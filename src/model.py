import torch
from PIL import Image
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

from src.paths import DATA_DIR

MODELS = [[28 * 28, 128, 64, 10]]
# Valores de normalização do MNIST
MEAN = 0.1307
STD = 0.3081

EMINIST_BALANCED_VALUES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, model_idx=0):
        super().__init__()

        if not 0 <= model_idx < len(MODELS):
            raise ValueError(f"Índice de modelo inválido: {model_idx}")

        model_layers = MODELS[model_idx]

        self.flatten = nn.Flatten()

        # Lista para armazenar dinamicamente as camadas
        layers = []
        for i in range(len(model_layers) - 1):
            # Adiciona a camada Linear
            layers.append(nn.Linear(model_layers[i], model_layers[i + 1]))

            # Adiciona a ReLU em todas as camadas, exceto na última (saída)
            if i < len(model_layers) - 2:
                layers.append(nn.ReLU())

        # Empacota tudo em um Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Bloco 1
        # 1. Primeira camada convolucional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()  # 32x28x28

        # 2. Segunda camada convolucional
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()  # 64x28x28

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x14x14
        self.drop1 = nn.Dropout2d(0.25)

        # Bloco 2
        # Terceira convulacional
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()  # 128x14x14

        # Quarta convulacional
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()  # 128x14x14

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x7x7
        self.drop2 = nn.Dropout2d(0.25)

        # Bloco 3
        self.flatten = nn.Flatten()  # 6272

        # 3. Camadas completamente conectadas (Dense / Linear)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 47)

    def forward(self, x):
        # Bloco 1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(self.relu2(self.bn2(self.conv2(x)))))

        # Bloco 2
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(self.relu4(self.bn4(self.conv4(x)))))

        # Classificação Final
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_drop(self.fc1_relu(self.fc1_bn(x)))
        x = self.fc2(x)

        return x


def custom_noise_erasing(img):
    if torch.rand(1).item() > 0.1:  # Probabilidade p=0.1
        return img

    # Calcula parâmetros da região cortada
    i, j, h, w, _ = v2.RandomErasing.get_params(
        img, scale=(0.01, 0.2), ratio=(0.2, 2.2), value=[0]
    )

    # Valores normalizados para preto (0.0) e cinza médio (0.5)
    val_black = (0.0 - MEAN) / STD  # ~ -0.424
    val_gray = (0.5 - MEAN) / STD  # ~  1.200

    # Gera ruído uniforme entre preto e cinza médio
    noise = torch.empty(
        (img.shape[0], h, w), dtype=img.dtype, device=img.device
    ).uniform_(val_black, val_gray)

    # Aplica o apagamento usando a API v2
    return v2.functional.erase(img, i, j, h, w, noise)


def fix_emnist_orientation(img):
    return img.rotate(-90).transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def get_transformers():
    degrees = 10.0

    # Transformação para converter as imagens em tensores e normaliza-las
    # Adicionado também um random de posição, rotação e escala
    # Além de um custom noise erasing
    return v2.Compose(
        [
            v2.Lambda(fix_emnist_orientation),
            v2.RandomAffine(
                (-degrees, degrees), translate=(0.1, 0.1), scale=(0.8, 1.1)
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((MEAN,), (STD,)),
            v2.Lambda(custom_noise_erasing),
        ]
    )


def get_emnist_dataset(train: bool, download: bool = False):
    return datasets.EMNIST(
        DATA_DIR,
        "balanced",
        train=train,
        download=download,
        transform=get_transformers(),
    )
