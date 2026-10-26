import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.neural_network.config import init_config
from src.neural_network.model import (
    SimpleCNN,
    get_emnist_dataset,
    load_checkpoint_weights,
)
from src.utils.paths import checkpoint_path


def main():
    torch.backends.cudnn.benchmark = True

    # 1. Initialize configuration.
    args, device = init_config()

    # 2. Download and load the dataset.
    train_dataset = get_emnist_dataset(True)
    print("2. Dataset loaded successfully!")

    # 3. Create a data loader to split the data into batches.
    batch_size = 64
    workers = 1
    if device.type == "cuda":
        batch_size = 256
        workers = 4

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=workers,
    )
    print(f"3. Loaded {len(train_loader)} batches successfully!")

    # 4. Create the network.
    # model = SimpleNeuralNetwork(args.input)
    model = SimpleCNN(num_classes=len(train_dataset.classes))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    # Training loop.
    epochs = 3
    print("6. Starting training...")

    start_total = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        batches_in_interval = 0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader, 0):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Run the network prediction.
            outputs = model(images)

            # Calculate the loss.
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss.
            loss.backward()

            # Update the weights and clear the gradients.
            optimizer.step()
            optimizer.zero_grad()

            running_loss += float(loss.item())
            batches_in_interval += 1

            # Calculate accuracy.
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            if batch_idx % 200 == 199 or batch_idx == len(train_loader) - 1:
                avg_loss = running_loss / batches_in_interval
                accuracy = correct / total
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"\t=> Epoch [{epoch + 1}/{epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy * 100:.2f}%"
                )
                # Reset the interval counters.
                running_loss = 0.0
                batches_in_interval = 0
                correct = 0
                total = 0

        epoch_duration = time.perf_counter() - epoch_start
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] -> Epoch {epoch + 1} finished in {epoch_duration:.2f}s"
        )

    total_duration = time.perf_counter() - start_total
    print(f"7. Training completed in {total_duration:.2f}s!")

    if args.output != 0:
        file = checkpoint_path(args.output)

    file.parent.mkdir(parents=True, exist_ok=True)
    content = {"model": model.state_dict(), "classes": train_dataset.classes}
    torch.save(content, file)
    print(f"8. Model saved successfully to {file}!")


if __name__ == "__main__":
    main()
