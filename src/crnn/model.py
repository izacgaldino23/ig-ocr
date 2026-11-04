from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torchvision.transforms import v2
from webdataset.compat import WebDataset

from src.utils.paths import checkpoint_path

ALPHABET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "áàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ.,!?-:;()\"'/#$ %"
)
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(ALPHABET)}
IDX2CHAR = {idx + 1: char for idx, char in enumerate(ALPHABET)}
BLANK_IDX = 0  # Blank from CTC character index


class OCRDataset(Dataset):
    def __init__(self, dataset_dir, train=True, img_height=32) -> None:
        self.dataset_dir = dataset_dir / ("train" if train else "test")
        self.img_height = img_height

        shards = sorted(self.dataset_dir.glob("shard-*.tar"))
        if not shards:
            raise ValueError(f"Base WebDataset não encontrada em {self.dataset_dir}")

        self.samples = list(
            WebDataset(
                [f"file:{shard.resolve().as_posix()}" for shard in shards],
                shardshuffle=False,
            )
        )
        if not self.samples:
            raise ValueError(f"Base WebDataset sem amostras em {self.dataset_dir}")

        self.num_samples = len(self.samples)
        self.data = [
            f"{Path(sample['__url__']).name}/{sample['__key__']}.png"
            for sample in self.samples
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_bytes = sample["png"]
        label = sample["txt"].decode("utf-8")

        # Decode memory buffer directly to Grayscale via OpenCV
        import numpy as np

        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

        if img is None or img.size == 0:
            raise FileNotFoundError(
                f"Error decoding image for index {idx} (label [{label}])"
            )

        # Proportional img resize keeping the ratio
        h, w = img.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions ({h}x{w}) for {idx}")

        new_w = max(1, int(w * (self.img_height / h)))
        img = cv2.resize(img, (int(new_w), int(self.img_height)))

        # Right padding to standardize width to max_width
        padded_img = torch.zeros((1, self.img_height, new_w))
        img_tensor = torch.from_numpy(img).float() / 255.0
        padded_img[0, :, :new_w] = img_tensor

        # Convert text to numerical indices
        target = [CHAR2IDX[c] for c in label if c in CHAR2IDX]
        target_length = len(target)

        return padded_img, torch.tensor(target, dtype=torch.long), target_length


class CRNN(nn.Module):
    def __init__(self, num_classes=104, name="1.0") -> None:
        super().__init__()

        self.name = name

        # 1. Feature Extractor (CNN)
        # Input: (B, 1, 32, img_width) -> Reduce height to 1 e width to img_width / 4 (4x width reducing)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 64, 16, 128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 128, 8, 64)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(8, 1), stride=(8, 1)
            ),  # Reduce height to 1: (B, 256, 1, W) -> T = W (img_width / 4) slices
        )

        # 2. Sequence Modeling (RNN)
        self.rnn = nn.LSTM(
            input_size=256, hidden_size=128, bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(0.2)

        # 3. Output Projection
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x, input_lengths=None):
        features = self.cnn(x)
        features = features.squeeze(2)  # (B, 256, W)
        features = features.permute(0, 2, 1)  # (B, W, 256) -> (Batch, Time, Feat)
        features = self.dropout(features)

        if input_lengths is None:
            out, _ = self.rnn(features)
        else:
            packed = pack_padded_sequence(
                features,
                input_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.shape[1]
            )

        logits = self.fc(out)  # (B, 64, num_classes)
        return logits

    def load_if_exists(self) -> bool:
        file = checkpoint_path(self.name, "ocr")
        if file.exists():
            content = torch.load(file, weights_only=True)

            self.load_state_dict(content["model"])
            print(f"Model {self.name} was loaded")
            return True

        print(f"Model {self.name} doesn't exist")
        return False

    def save_model(self, output_name=""):
        name = self.name if output_name == "" else output_name
        file = checkpoint_path(name, "ocr")
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict()}, file)

        return name


gpu_augmentations = v2.Compose(
    [
        v2.RandomApply(
            [
                v2.RandomAffine(
                    degrees=(-2, 2),
                    scale=(0.95, 1.05),
                    shear=(-2, 2),
                    interpolation=v2.InterpolationMode.BILINEAR,
                    fill=1.0,
                )
            ],
            p=0.7,
        ),
        v2.RandomApply([v2.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7))], p=0.15),
    ]
)


def ocr_collate_fn(batch):
    # batch: list of tuples (img_tensor, target_tensor, target_length)
    images, targets, target_lengths = zip(*batch)
    image_widths = torch.tensor([img.shape[2] for img in images])
    input_lengths = torch.tensor([img.shape[2] // 4 for img in images])

    # 1. Finds the maximum width within THIS specific batch
    max_w = max(img.shape[2] for img in images)

    # 2. Applies right-padding to each image until reaching max_w
    padded_images = []
    for img in images:
        _, w = img.shape[1], img.shape[2]
        pad_w = max_w - w
        # F.pad applies (left, right, top, bottom)
        # padded_img = F.pad(img, (0, pad_w, 0, 0), value=1.0)
        padded_img = F.pad(img, (0, pad_w, 0, 0), mode="replicate")
        padded_images.append(padded_img)

    images_tensor = torch.stack(padded_images, dim=0)
    targets_tensor = torch.cat(targets, dim=0)

    return (
        images_tensor,
        targets_tensor,
        torch.tensor(target_lengths),
        input_lengths,
        image_widths,
    )
