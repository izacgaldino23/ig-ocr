import time
from contextlib import nullcontext
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.neural_network.config import init_config
from src.neural_network.model import (
    COMPACT_CNN_ARCHITECTURE,
    COMPACT_CNN_CHANNELS,
    DEFAULT_CNN_CHANNELS,
    MEAN,
    SIMPLE_CNN_ARCHITECTURE,
    STD,
    SimpleCNN,
    fix_emnist_orientation,
    get_emnist_dataset,
    get_gpu_augmentations,
    load_checkpoint_weights,
)
from src.utils.paths import checkpoint_path


def main():
    torch.backends.cudnn.benchmark = True

    # 1. Initialize configuration.
    args, device = init_config()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # 2. Download and load the dataset.
    if args.gpu_uint8 and device.type != "cuda":
        raise ValueError("--gpu-uint8 requires CUDA")

    train_dataset = get_emnist_dataset(True, apply_transform=not args.gpu_uint8)
    print("2. Dataset loaded successfully!")

    # 3. Create a data loader to split the data into batches.
    default_batch_size = 64
    default_workers = 1
    if device.type == "cuda":
        default_batch_size = 1024
        default_workers = 4
    batch_size = args.batch_size or default_batch_size
    workers = args.workers if args.workers is not None else default_workers
    training_images: torch.Tensor | None = None
    training_labels: torch.Tensor | None = None
    train_loader: DataLoader | None = None
    if args.gpu_uint8:
        if train_dataset.data.dtype != torch.uint8:
            raise TypeError(
                f"Expected raw EMNIST images as uint8, got {train_dataset.data.dtype}"
            )
        preload_start = time.perf_counter()
        training_images = (
            fix_emnist_orientation(train_dataset.data.to(device, non_blocking=True))
            .unsqueeze(1)
            .contiguous()
        )
        training_labels = train_dataset.targets.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        preload_duration = time.perf_counter() - preload_start
        print(
            f"3. Preloaded {len(train_dataset)} uint8 images on CUDA "
            f"in {preload_duration:.2f}s."
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        print(f"3. Loaded {len(train_loader)} batches successfully!")
    num_batches = (len(train_dataset) + batch_size - 1) // batch_size
    if args.gpu_uint8:
        print(f"3. Training will use {num_batches} batches per epoch.")

    # 4. Create the network.
    # model = SimpleNeuralNetwork(args.input)
    channels = COMPACT_CNN_CHANNELS if args.compact else DEFAULT_CNN_CHANNELS
    architecture = COMPACT_CNN_ARCHITECTURE if args.compact else SIMPLE_CNN_ARCHITECTURE
    model = SimpleCNN(num_classes=len(train_dataset.classes), channels=channels)
    file = checkpoint_path(args.input)
    print("4. Network created successfully!")

    if file.exists():
        content = torch.load(file)
        if load_checkpoint_weights(
            model, content["model"], content["classes"], train_dataset.classes
        ):
            print(
                "5. Compatible weights loaded; new classes will be trained from scratch."
            )
        else:
            print("5. Incompatible architecture; starting a new model.")
            if args.output == 0 or args.output == args.input:
                raise ValueError(
                    "The input checkpoint is incompatible with the ByClass model. "
                    "Choose a different output to preserve the existing checkpoint."
                )
    else:
        print("5. New model!")

    model = model.to(device)
    # 1. Count the number of images per class in the dataset.
    # Use the PyTorch targets.
    targets = train_dataset.targets.detach().clone()
    class_counts = torch.bincount(targets).float()

    # 2. Calculate weights inversely proportional to each class frequency.
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = torch.clamp(class_weights, max=100.0)

    # Normalize the weights to prevent excessively large loss values.
    class_weights = class_weights / class_weights.mean()

    # Move the weight tensor to the same GPU or CPU as the model.
    class_weights = class_weights.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop.
    epochs = 3
    print("6. Starting training...")

    start_total = time.perf_counter()

    gpu_transforms = get_gpu_augmentations().to(device)

    for epoch in range(epochs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] -> Epoch {epoch + 1} started")

        epoch_start = time.perf_counter()
        running_loss = torch.zeros((), device=device)
        batches_in_interval = 0
        total = 0
        correct = torch.zeros((), device=device, dtype=torch.long)

        prof_context = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./log/profiler"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            if args.profile
            else nullcontext()
        )

        with prof_context as prof:
            if args.gpu_uint8:
                if training_images is None or training_labels is None:
                    raise RuntimeError("GPU uint8 dataset was not initialized")
                indices = torch.randperm(len(train_dataset)).to(
                    device, non_blocking=True
                )
                epoch_batches = (
                    (
                        training_images[indices[start : start + batch_size]],
                        training_labels[indices[start : start + batch_size]],
                    )
                    for start in range(0, len(train_dataset), batch_size)
                )
            else:
                if train_loader is None:
                    raise RuntimeError("DataLoader was not initialized")
                epoch_batches = iter(train_loader)

            for batch_idx, (images, labels) in enumerate(epoch_batches):
                if args.gpu_uint8:
                    images = images.to(torch.float32).mul_(1.0 / 255.0)
                    images.sub_(MEAN).div_(STD)
                else:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                if model.training:
                    images = gpu_transforms(images)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += loss.detach()
                batches_in_interval += 1

                # Calculate accuracy.
                total += labels.size(0)
                correct += (outputs.detach().argmax(dim=1) == labels).sum()

                if batch_idx % 200 == 199 or batch_idx == num_batches - 1:
                    avg_loss = running_loss.item() / batches_in_interval
                    accuracy = correct.item() / total
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"\t=> Epoch [{epoch + 1}/{epochs}], "
                        f"Batch [{batch_idx + 1}/{num_batches}], "
                        f"Loss: {avg_loss:.4f}, "
                        f"Accuracy: {accuracy * 100:.2f}%"
                    )
                    # Reset the interval counters.
                    running_loss.zero_()
                    batches_in_interval = 0
                    correct.zero_()
                    total = 0

                if prof is not None:
                    prof.step()

        epoch_duration = time.perf_counter() - epoch_start
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] -> Epoch {epoch + 1} finished in {epoch_duration:.2f}s"
        )

    total_duration = time.perf_counter() - start_total
    print(f"7. Training completed in {total_duration:.2f}s!")

    if args.output != 0:
        file = checkpoint_path(args.output)

    file.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "model": model.state_dict(),
        "classes": train_dataset.classes,
        "architecture": architecture,
    }
    torch.save(content, file)
    print(f"8. Model saved successfully to {file}!")


if __name__ == "__main__":
    main()
