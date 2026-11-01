import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.neural_network.model import (
    fix_emnist_orientation,
    get_chars74k_dataset,
    get_emnist_dataset,
)


def _stratified_split(labels, seed):
    labels = labels.cpu()
    class_indices = torch.bincount(labels)
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []

    for label, count in enumerate(class_indices.tolist()):
        indices = torch.where(labels == label)[0]
        if count < 2:
            raise ValueError(
                f"Class index {label} needs at least two samples for a train/validation split"
            )
        shuffled = indices[torch.randperm(count, generator=generator)]
        validation_count = min(count - 1, max(1, int(count * 0.1)))
        validation_indices.append(shuffled[:validation_count])
        train_indices.append(shuffled[validation_count:])

    return torch.cat(train_indices), torch.cat(validation_indices)


def prepare_training_data(args, device):
    if args.gpu_uint8 and device.type != "cuda":
        raise ValueError("--gpu-uint8 requires CUDA")

    fine_tuning = args.fine_tune is not None

    # if fine_tuning:
    #     train_dataset = get_chars74k_dataset()
    # else:
    #     train_dataset = get_emnist_dataset(True, apply_transform=not args.gpu_uint8)
    emnist_dataset = get_emnist_dataset(True, apply_transform=not args.gpu_uint8)
    chars74k_dataset = get_chars74k_dataset(not args.gpu_uint8)
    train_dataset = ConcatDataset([emnist_dataset, chars74k_dataset])
    # train_dataset = ConcatDataset([chars74k_dataset])
    print(
        f"2. Dataset loaded successfully! Using {'emnist' if not fine_tuning else 'chars74k'}"
    )
    train_indices, validation_indices = _stratified_split(
        get_dataset_targets(train_dataset), args.seed
    )

    # 3. Create a data loader to split the data into batches.
    default_batch_size = 64
    default_workers = 1
    if device.type == "cuda":
        default_batch_size = 1280
        default_workers = 4
    batch_size = args.batch_size or default_batch_size
    workers = args.workers if args.workers is not None else default_workers
    training_images: torch.Tensor | None = None
    training_labels: torch.Tensor | None = None
    train_indices_device: torch.Tensor | None = None
    validation_indices_device: torch.Tensor | None = None
    train_loader: DataLoader | None = None
    validation_loader: DataLoader | None = None
    if args.gpu_uint8:
        if get_dataset_data(train_dataset).dtype != torch.uint8:
            raise TypeError(
                f"Expected raw EMNIST images as uint8, got {get_dataset_data(train_dataset).dtype}"
            )
        preload_start = time.perf_counter()
        training_images = (
            fix_emnist_orientation(
                get_dataset_data(train_dataset).to(device, non_blocking=True)
            )
            .unsqueeze(1)
            .contiguous()
        )
        training_labels = get_dataset_targets(train_dataset).to(
            device, non_blocking=True
        )
        train_indices_device = train_indices.to(device, non_blocking=True)
        validation_indices_device = validation_indices.to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        preload_duration = time.perf_counter() - preload_start
        print(
            f"3. Preloaded {len(train_dataset)} uint8 images on CUDA "
            f"in {preload_duration:.2f}s."
        )
    else:
        train_subset = Subset(train_dataset, train_indices.tolist())
        validation_subset = Subset(train_dataset, validation_indices.tolist())
        train_loader = DataLoader(
            train_subset,
            shuffle=True,
            pin_memory=device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        validation_loader = DataLoader(
            validation_subset,
            shuffle=False,
            pin_memory=device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        print(
            f"3. Loaded {len(train_loader)} training and "
            f"{len(validation_loader)} validation batches."
        )
    train_count = len(train_indices)
    num_batches = (train_count + batch_size - 1) // batch_size
    num_validation_batches = (len(validation_indices) + batch_size - 1) // batch_size
    if args.gpu_uint8:
        print(
            f"3. Split {train_count} training and {len(validation_indices)} validation "
            f"samples; {num_batches} and {num_validation_batches} batches respectively."
        )
    return (
        train_dataset,
        train_indices,
        train_loader,
        validation_loader,
        training_images,
        training_labels,
        train_indices_device,
        validation_indices_device,
        batch_size,
        num_batches,
    )


def get_dataset_targets(combined_dataset):
    return torch.cat([d.targets for d in combined_dataset.datasets])


def get_dataset_data(combined_dataset):
    return torch.cat([d.data for d in combined_dataset.datasets])


def get_dataset_classes(combined_dataset):
    return combined_dataset.datasets[0].classes
