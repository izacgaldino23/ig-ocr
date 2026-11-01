import math
import os
import string
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms import v2
from torchvision.utils import _Image_fromarray

from src.neural_network.model import MEAN, STD
from src.utils.paths import DATA_DIR


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


def random_affine_batch(img):
    was_unbatched = img.ndim == 3
    images = img.unsqueeze(0) if was_unbatched else img
    batch_size, _, height, width = images.shape
    device = images.device

    angle = torch.empty(batch_size, device=device).uniform_(-15, 15)
    scale = torch.empty(batch_size, device=device).uniform_(0.8, 1.1)
    translate_x = (
        torch.empty(batch_size, device=device)
        .uniform_(-0.1 * width, 0.1 * width)
        .round()
    )
    translate_y = (
        torch.empty(batch_size, device=device)
        .uniform_(-0.1 * height, 0.1 * height)
        .round()
    )

    angle = torch.deg2rad(angle)
    cosine = angle.cos() / scale
    sine = angle.sin() / scale
    zero = torch.zeros_like(cosine)
    theta = torch.stack(
        (
            torch.stack((cosine, sine * (height / width), zero), dim=1),
            torch.stack((-sine * (width / height), cosine, zero), dim=1),
        ),
        dim=1,
    )
    theta[:, 0, 2] = -2 * (cosine * translate_x + sine * translate_y) / width
    theta[:, 1, 2] = 2 * (sine * translate_x - cosine * translate_y) / height

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    transformed = F.grid_sample(
        images,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    return transformed.squeeze(0) if was_unbatched else transformed


def custom_noise_erasing(img):
    was_unbatched = img.ndim == 3
    images = img.unsqueeze(0) if was_unbatched else img
    batch_size, _, height, width = images.shape
    device = images.device

    erase = torch.rand(batch_size, device=device) < 0.1
    area = height * width
    log_ratio_min = math.log(0.2)
    log_ratio_max = math.log(2.2)
    scale = torch.empty(batch_size, 10, device=device).uniform_(0.01, 0.2)
    ratio = (
        torch.empty(batch_size, 10, device=device)
        .uniform_(log_ratio_min, log_ratio_max)
        .exp_()
    )
    candidate_height = (area * scale * ratio).sqrt().round().long()
    candidate_width = (area * scale / ratio).sqrt().round().long()
    valid = (
        erase[:, None]
        & (candidate_height > 0)
        & (candidate_height < height)
        & (candidate_width > 0)
        & (candidate_width < width)
    )
    has_valid = valid.any(dim=1)
    first_valid = valid.to(torch.long).argmax(dim=1, keepdim=True)
    erase_height = candidate_height.gather(1, first_valid).squeeze(1)
    erase_width = candidate_width.gather(1, first_valid).squeeze(1)
    erase_height = torch.where(has_valid, erase_height, height)
    erase_width = torch.where(has_valid, erase_width, width)

    top = (torch.rand(batch_size, device=device) * (height - erase_height + 1)).long()
    left = (torch.rand(batch_size, device=device) * (width - erase_width + 1)).long()

    rows = torch.arange(height, device=device).view(1, height, 1)
    columns = torch.arange(width, device=device).view(1, 1, width)
    region = (
        erase[:, None, None]
        & (rows >= top[:, None, None])
        & (rows < (top + erase_height)[:, None, None])
        & (columns >= left[:, None, None])
        & (columns < (left + erase_width)[:, None, None])
    )

    val_black = (0.0 - MEAN) / STD
    val_gray = (0.5 - MEAN) / STD
    noise = torch.empty_like(images).uniform_(val_black, val_gray)
    erased_images = torch.where(region.unsqueeze(1), noise, images)
    return erased_images.squeeze(0) if was_unbatched else erased_images


def fix_emnist_orientation(img):
    # return img.rotate(-90).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img = torch.rot90(img, k=-1, dims=(-2, -1))
    return torch.flip(img, dims=[-1])


def get_transformers():
    # Convert images to tensors and normalize them.
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(fix_emnist_orientation),
            v2.Normalize((MEAN,), (STD,)),
        ]
    )


def get_gpu_augmentations():
    return v2.Compose(
        [
            # v2.RandomAffine((-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.1)),
            v2.Lambda(random_affine_batch),
            v2.ElasticTransform(alpha=50.0, sigma=5.0),
            # v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # v2.Lambda(custom_noise_erasing),
        ]
    )


def get_emnist_dataset(
    train: bool, download: bool = False, apply_transform: bool = True
):
    return datasets.EMNIST(
        DATA_DIR,
        "byclass",
        train=train,
        download=download,
        transform=get_transformers() if apply_transform else None,
    )


def get_chars74k_dataset(apply_transform: bool = True):
    transform = get_chars74k_transformers() if apply_transform else None
    return Chars74KFontDataset(data_dir=DATA_DIR, transform=transform)


class Chars74KFontDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.processed_dir = data_dir / "processed_chars74k"
        self.transform = transform

        images_file = os.path.join(self.processed_dir, "chars74k-fnt-images-idx3-ubyte")
        labels_file = os.path.join(self.processed_dir, "chars74k-fnt-labels-idx1-ubyte")
        self.classes = string.digits + string.ascii_uppercase + string.ascii_lowercase

        if not (os.path.exists(images_file) and os.path.exists(labels_file)):
            raise FileNotFoundError(
                f"IDX files not found in {self.processed_dir}. "
                "Run 'process_and_save_chars74k()' first."
            )

        # Load binary files into memory (returns torch.uint8 Tensors)
        self.data: torch.Tensor = read_image_file(images_file)  # Shape: [N, 28, 28]
        self.targets: torch.Tensor = read_label_file(labels_file)  # Shape: [N]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img, target = self.data[idx], int(self.targets[idx])

        img = _Image_fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_chars74k_transformers():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(fix_emnist_orientation),
            # v2.Resize([26, 26]),
            # v2.Pad(1, 0.0),
            v2.Normalize((MEAN,), (STD,)),
        ]
    )
