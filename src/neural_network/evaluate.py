import torch
from torch.utils.data import DataLoader

from src.neural_network.config import init_config
from src.neural_network.model import get_emnist_dataset, model_from_checkpoint
from src.utils.paths import checkpoint_path


def main():
    args, device = init_config()

    test_dataset = get_emnist_dataset(False)
    print("Dataset loaded successfully!")

    default_batch_size = 256 if device.type == "cuda" else 64
    default_workers = 4 if device.type == "cuda" else 1
    batch_size = args.batch_size or default_batch_size
    workers = args.workers if args.workers is not None else default_workers
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=workers,
        persistent_workers=workers > 0,
    )

    file = checkpoint_path(args.input)
    content = torch.load(file, map_location="cpu", weights_only=True)
    class_names: list[str] = content["classes"]
    model = model_from_checkpoint(content).to(device)
    model.eval()

    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.long, device=device
    )

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            batch_confusion = torch.bincount(
                labels * len(class_names) + predicted,
                minlength=len(class_names) ** 2,
            ).reshape(len(class_names), len(class_names))
            confusion += batch_confusion

    accuracy = correct.item() / total
    print(f"Test dataset accuracy: {accuracy * 100:.2f}%")

    confusion = confusion.cpu()
    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diagonal()
    class_accuracy = class_correct.float() / class_totals.clamp_min(1)
    for i in class_accuracy.argsort(descending=True).tolist():
        print(
            f"{class_names[i]}: {class_accuracy[i]:.4f}. "
            f"[{class_correct[i]} of {class_totals[i]}]"
        )


if __name__ == "__main__":
    main()
