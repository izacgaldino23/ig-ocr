import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_dtype
from torchvision.utils import save_image

from src.neural_network.config import init_config
from src.neural_network.data import fix_emnist_orientation, get_chars74k_dataset
from src.neural_network.model import (
    MEAN,
    STD,
    model_from_checkpoint,
)
from src.utils.paths import PROJECT_ROOT, checkpoint_path


def _gpu_uint8_batches(test_dataset, batch_size, device):
    if test_dataset.data.dtype != torch.uint8:
        raise TypeError(
            f"Expected raw EMNIST images as uint8, got {test_dataset.data.dtype}"
        )

    preload_start = time.perf_counter()
    images = (
        fix_emnist_orientation(test_dataset.data.to(device, non_blocking=True))
        .unsqueeze(1)
        .contiguous()
    )
    labels = test_dataset.targets.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    print(
        f"Preloaded {len(test_dataset)} uint8 test images on CUDA "
        f"in {time.perf_counter() - preload_start:.2f}s."
    )

    for start in range(0, len(test_dataset), batch_size):
        batch_images = images[start : start + batch_size].to(torch.float32)
        batch_images.mul_(1.0 / 255.0).sub_(MEAN).div_(STD)
        yield batch_images, labels[start : start + batch_size]


def _evaluate(model, test_loader, class_names, device, writer):
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.long, device=device
    )

    with torch.inference_mode():
        for image, label in test_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            outputs = model(image)
            predicted = outputs.argmax(dim=1)
            total += label.size(0)
            correct += (predicted == label).sum()
            batch_confusion = torch.bincount(
                label * len(class_names) + predicted,
                minlength=len(class_names) ** 2,
            ).reshape(len(class_names), len(class_names))
            confusion += batch_confusion

    accuracy = correct.item() / total
    print(f"Test dataset accuracy: {accuracy * 100:.2f}%")

    confusion = confusion.cpu()
    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diagonal()
    class_accuracy = class_correct.float() / class_totals.clamp_min(1)
    # macro_accuracy = class_accuracy.mean().item()
    # writer.add_scalar("Accuracy/test", accuracy)
    # writer.add_scalar("Accuracy/macro_test", macro_accuracy)
    for i in class_accuracy.argsort(descending=True).tolist():
        # writer.add_scalar(f"Accuracy/test_by_class/{class_names[i]}", class_accuracy[i])
        print(
            f"{class_names[i]}: {class_accuracy[i]:.4f}. "
            f"[{class_correct[i]} of {class_totals[i]}]"
        )


def main():
    args, device = init_config()
    if args.gpu_uint8 and device.type != "cuda":
        raise ValueError("--gpu-uint8 requires CUDA")

    # test_dataset = get_emnist_dataset(False, True)
    test_dataset = get_chars74k_dataset(not args.gpu_uint8)
    print("Dataset loaded successfully!")

    # img = test_dataset.data[5000]
    # img_display = img.squeeze(0) * STD + MEAN

    # img = v2.functional.to_pil_image(img_display)
    # img.save(PROJECT_ROOT / "data" / "imagem_emnist.png")
    # print(test_dataset.targets[5000].item())
    # plt.imshow(img_display, interpolation="nearest")
    # plt.show()
    # return

    default_batch_size = 1024 if device.type == "cuda" else 64
    default_workers = 4 if device.type == "cuda" else 1
    batch_size = args.batch_size or default_batch_size
    if args.gpu_uint8:
        test_loader = _gpu_uint8_batches(test_dataset, batch_size, device)
    else:
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
    # run_name = f"{datetime.now():%Y%m%d-%H%M%S}-eval-{file.stem}"
    # with SummaryWriter(log_dir=f"log/runs/{run_name}") as writer:
    _evaluate(model, test_loader, class_names, device, None)


if __name__ == "__main__":
    main()
