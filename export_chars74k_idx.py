import os
import string
import struct
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import v2

from src.utils.paths import PROJECT_ROOT


def write_idx3_ubyte(images: torch.Tensor, filename: str):
    """Saves a uint8 Tensor of shape [N, 28, 28] to IDX3-ubyte format."""
    num_images, rows, cols = images.shape
    magic_number = 2051  # 0x00000803 (3D tensor of uint8)

    with open(filename, "wb") as f:
        f.write(struct.pack(">IIII", magic_number, num_images, rows, cols))
        f.write(images.numpy().astype(np.uint8).tobytes())


def write_idx1_ubyte(targets: torch.Tensor, filename: str):
    """Saves a int64/long Tensor of shape [N] to IDX1-ubyte format."""
    num_items = len(targets)
    magic_number = 2049  # 0x00000801 (1D tensor of uint8)

    with open(filename, "wb") as f:
        f.write(struct.pack(">II", magic_number, num_items))
        f.write(targets.numpy().astype(np.uint8).tobytes())


def process_and_save_chars74k(data_dir: str, output_dir: str):
    root_path = Path(data_dir) / "English" / "Fnt"
    os.makedirs(output_dir, exist_ok=True)

    if not root_path.exists():
        raise FileNotFoundError(f"Directory {root_path} not found.")

    raw_images = []
    raw_labels = []

    print("Processing Chars74K images...")
    classes = string.digits + string.ascii_uppercase + string.ascii_lowercase
    data = np.array([], dtype=np.uint8)
    for sample_folder in sorted(root_path.glob("Sample*")):
        sample_num = int(sample_folder.name.replace("Sample", ""))
        label = sample_num - 1  # Map 1-62 to 0-61 (EMNIST ByClass layout)

        samples = 0
        print(f"Processing {classes[label]} ...")
        for img_path in sample_folder.glob("*.png"):
            img_rgba = Image.open(img_path).convert("RGBA")
            bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            img_composite = Image.alpha_composite(bg, img_rgba).convert("L")

            img_22x22 = img_composite.resize((22, 22), Image.Resampling.LANCZOS)

            img_arr = np.array(img_22x22, dtype=np.uint8)

            img_resized = np.pad(img_arr, 3, mode="edge")
            img_inverted = 255 - img_resized
            img_inverted = img_inverted.T

            img_inverted[img_inverted < 50] = 0
            # data = img_inverted
            # plt.imshow(data, interpolation="nearest")
            # plt.show()
            # return

            raw_images.append(img_inverted)
            raw_labels.append(label)

            samples += 1

        # break

        print(f"Processed {classes[label]} with {samples} items!")

    # img = v2.functional.to_pil_image(data)
    # img.save(PROJECT_ROOT / "data" / "image_chars74k.png")
    # return

    # Convert lists to Tensors
    data_tensor = torch.from_numpy(np.array(raw_images, dtype=np.uint8))
    targets_tensor = torch.tensor(raw_labels, dtype=torch.long)

    # Save to IDX files in output_dir
    images_file = os.path.join(output_dir, "chars74k-fnt-images-idx3-ubyte")
    labels_file = os.path.join(output_dir, "chars74k-fnt-labels-idx1-ubyte")

    print("Saving to IDX format...")
    write_idx3_ubyte(data_tensor, images_file)
    write_idx1_ubyte(targets_tensor, labels_file)

    print(f"Done! {len(targets_tensor)} samples saved to: {output_dir}")


if __name__ == "__main__":
    process_and_save_chars74k(
        data_dir="./data/raw", output_dir="./data/raw/processed_chars74k"
    )
