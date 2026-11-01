import torch
from torch import nn

MODELS = [[28 * 28, 128, 64, 10]]
# MNIST normalization values.
MEAN = 0.1736
STD = 0.3317


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, model_idx=0):
        super().__init__()

        if not 0 <= model_idx < len(MODELS):
            raise ValueError(f"Invalid model index: {model_idx}")

        model_layers = MODELS[model_idx]

        self.flatten = nn.Flatten()

        # Store the layers dynamically.
        layers = []
        for i in range(len(model_layers) - 1):
            # Add a linear layer.
            layers.append(nn.Linear(model_layers[i], model_layers[i + 1]))

            # Add ReLU to every layer except the final output layer.
            if i < len(model_layers) - 2:
                layers.append(nn.ReLU())

        # Package the layers in a Sequential module.
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


DEFAULT_CNN_CHANNELS = (32, 64, 128, 128)
COMPACT_CNN_CHANNELS = (24, 48, 96, 96)
SIMPLE_CNN_ARCHITECTURE = "simple_cnn_v1"
COMPACT_CNN_ARCHITECTURE = "simple_cnn_compact_v1"


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 47,
        channels: tuple[int, int, int, int] = DEFAULT_CNN_CHANNELS,
    ):
        super().__init__()
        if len(channels) != 4 or any(channel < 1 for channel in channels):
            raise ValueError("SimpleCNN requires four positive channel sizes")
        self.channels = channels
        channel1, channel2, channel3, channel4 = channels

        # Block 1
        # 1.1 First convolutional layer.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=channel1, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channel1)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1.2 Second convolutional layer.
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel2)
        self.relu2 = nn.ReLU()

        self.drop1 = nn.Dropout2d(0.1)

        # Block 2
        # 2.1 Third convolutional layer.
        self.conv3 = nn.Conv2d(channel2, channel3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channel3)
        self.relu3 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2.2 Fourth convolutional layer.
        self.conv4 = nn.Conv2d(channel3, channel4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(channel4)
        self.relu4 = nn.ReLU()

        self.drop2 = nn.Dropout2d(0.1)

        # Block 3
        # 3.1 Flatten data to one dimension
        self.flatten = nn.Flatten()

        # 3.2 Fully connected layers (Dense / Linear).
        self.fc1 = nn.Linear(channel4 * 7 * 7, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_relu = nn.ReLU()

        self.fc1_drop = nn.Dropout(0.4)

        # 3.3 Linear layer to outputs
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop1(self.relu2(self.bn2(self.conv2(x))))

        # Block 2
        x = self.pool2(self.relu3(self.bn3(self.conv3(x))))
        x = self.drop2(self.relu4(self.bn4(self.conv4(x))))

        # Final classification.
        x = self.flatten(x)
        x = self.fc1_bn(self.fc1(x))
        x = self.fc1_drop(self.fc1_relu(x))
        x = self.fc2(x)

        return x


def load_checkpoint_weights(
    model: SimpleCNN,
    state: dict[str, torch.Tensor],
    checkpoint_classes: list[str],
    target_classes: list[str],
) -> bool:
    model_state = model.state_dict()
    output_keys = {"fc2.weight", "fc2.bias"}
    if any(name not in state for name in model_state):
        return False

    if (
        state["fc2.weight"].shape[0] != len(checkpoint_classes)
        or state["fc2.bias"].shape[0] != len(checkpoint_classes)
        or state["fc2.weight"].shape[1] != model_state["fc2.weight"].shape[1]
        or state["fc2.bias"].shape != model_state["fc2.bias"].shape[:1]
    ):
        return False

    source_channels = tuple(state[f"conv{idx}.weight"].shape[0] for idx in range(1, 5))
    if any(
        state[f"conv{idx}.weight"].shape[1]
        != (1 if idx == 1 else source_channels[idx - 2])
        for idx in range(1, 5)
    ):
        return False

    adapted_state = {}
    for name, target in model_state.items():
        if name in output_keys:
            continue

        source = state[name].to(device=target.device, dtype=target.dtype)
        if source.shape == target.shape:
            adapted_state[name] = source
        elif name.startswith("conv") and name.endswith(".weight"):
            if (
                source.ndim != 4
                or target.ndim != 4
                or source.shape[2:] != target.shape[2:]
                or any(
                    target_size > source_size
                    for target_size, source_size in zip(
                        target.shape[:2], source.shape[:2]
                    )
                )
            ):
                return False
            adapted_state[name] = source[: target.shape[0], : target.shape[1]]
        elif (
            name.startswith("conv")
            and name.endswith(".bias")
            or name.startswith("bn")
            and source.ndim == target.ndim == 1
        ):
            if source.ndim != 1 or target.ndim != 1 or target.numel() > source.numel():
                return False
            adapted_state[name] = source[: target.numel()]
        elif name == "fc1.weight":
            if source.ndim != 2 or target.ndim != 2:
                return False
            source_channels_last = source.shape[1] // 49
            target_channels_last = target.shape[1] // 49
            if (
                source.shape[0] != target.shape[0]
                or source.shape[1] != source_channels_last * 49
                or target.shape[1] != target_channels_last * 49
                or source_channels_last != source_channels[-1]
                or target_channels_last != model.channels[-1]
                or target_channels_last > source_channels_last
            ):
                return False
            adapted_state[name] = source.reshape(
                source.shape[0], source_channels_last, 49
            )[:, :target_channels_last].reshape_as(target)
        else:
            return False

    model_state.update(adapted_state)

    with torch.no_grad():
        for old_idx, class_name in enumerate(checkpoint_classes):
            if class_name in target_classes:
                new_idx = target_classes.index(class_name)
                model_state["fc2.weight"][new_idx].copy_(
                    state["fc2.weight"][old_idx].to(model_state["fc2.weight"].device)
                )
                model_state["fc2.bias"][new_idx].copy_(
                    state["fc2.bias"][old_idx].to(model_state["fc2.bias"].device)
                )

    model.load_state_dict(model_state)
    return True


def model_from_checkpoint(content: dict) -> SimpleCNN:
    class_names: list[str] = content["classes"]
    architecture = content.get("architecture", SIMPLE_CNN_ARCHITECTURE)
    if architecture == SIMPLE_CNN_ARCHITECTURE:
        channels = DEFAULT_CNN_CHANNELS
    elif architecture == COMPACT_CNN_ARCHITECTURE:
        channels = COMPACT_CNN_CHANNELS
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")

    model = SimpleCNN(num_classes=len(class_names), channels=channels)
    model.load_state_dict(content["model"])
    return model
