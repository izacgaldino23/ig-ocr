import torch
from torch.utils.data import DataLoader

from src.neural_network.config import init_config
from src.neural_network.model import SimpleCNN, get_emnist_dataset


def calculate_weights(class_counts: torch.Tensor, strategy: str = "log"):
    if strategy == "log":
        weights = 1.0 / torch.log(torch.tensor(1.2) + class_counts)
    elif strategy == "sqrt":
        weights = 1.0 / (class_counts**0.5)
    elif strategy == "power_0.7":
        weights = 1.0 / (class_counts**0.7)
        weights = torch.clamp(weights, max=50.0)
    elif strategy == "inverse":
        weights = 1.0 / (class_counts + 1e-5)
        weights = torch.clamp(weights, max=100.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize by the mean to keep the loss scale stable.
    weights = weights / weights.mean()
    return weights


def main():
    print("Starting a quick class-weight test...")
    _, device = init_config()

    # Load the training dataset.
    train_dataset = get_emnist_dataset(True)
    class_names = train_dataset.classes
    targets = train_dataset.targets.detach().clone()
    class_counts = torch.bincount(targets).float()

    strategies = ["log", "sqrt", "power_0.7", "inverse"]

    _, sorted_indices = torch.sort(class_counts)
    print("\n--- 5 Rarest Classes ---")
    for i in range(5):
        idx = int(sorted_indices[i].item())
        print(f"  {class_names[idx]}: {int(class_counts[idx].item())} samples")

    print("\n--- 5 Most Common Classes ---")
    for i in range(5):
        idx = int(sorted_indices[-(i + 1)].item())
        print(f"  {class_names[idx]}: {int(class_counts[idx].item())} samples")
    print()

    most_common_idx = int(sorted_indices[-1].item())
    most_rare_idx = int(sorted_indices[0].item())

    for strat in strategies:
        weights = calculate_weights(class_counts, strat)

        w_common = int(weights[most_common_idx].item())
        w_rare = int(weights[most_rare_idx].item())
        ratio = w_rare / w_common

        print(f"Strategy [{strat}]:")
        print(
            f"  - Common class weight ({class_names[most_common_idx]}): {w_common:.4f}"
        )
        print(f"  - Rare class weight ({class_names[most_rare_idx]}): {w_rare:.4f}")
        print(f"  - Ratio (Rare / Common): {ratio:.1f}x")
        print(f"  - Min / Max weights: {weights.min():.4f} / {weights.max():.4f}")

        # Test the loss with a real batch.
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        images, labels = next(iter(train_loader))

        model = SimpleCNN(num_classes=len(class_names)).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))

        with torch.no_grad():
            outputs = model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))

        print(f"  - Initial loss for one batch: {loss.item():.4f}\n")


if __name__ == "__main__":
    main()
