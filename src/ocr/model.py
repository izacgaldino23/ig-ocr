import os
import tarfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.utils.paths import checkpoint_path

ALPHABET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "áàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ.,!?-:;()\"'/#$ %"
    "ºª–—“”‘’"
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

        self.samples = []
        self.data = []
        self.widths = []
        self._shard_files = {}
        self._pid = os.getpid()

        for shard in shards:
            samples = {}
            with tarfile.open(shard, mode="r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    key, separator, extension = member.name.rpartition(".")
                    if separator and extension in {"png", "txt"}:
                        samples.setdefault(key, {})[extension] = (
                            member.offset_data,
                            member.size,
                        )

            with open(shard, "rb") as shard_file:
                for key, members in samples.items():
                    if "png" not in members or "txt" not in members:
                        raise ValueError(f"Amostra incompleta '{key}' no shard {shard}")
                    image_offset, image_size = members["png"]
                    label_offset, label_size = members["txt"]
                    shard_file.seek(image_offset)
                    png_header = shard_file.read(24)
                    if (
                        len(png_header) != 24
                        or png_header[:8] != b"\x89PNG\r\n\x1a\n"
                        or png_header[12:16] != b"IHDR"
                    ):
                        raise ValueError(f"Imagem PNG inválida em {shard}:{key}")
                    width = int.from_bytes(png_header[16:20], "big")
                    height = int.from_bytes(png_header[20:24], "big")
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Dimensões inválidas em {shard}:{key}")

                    self.samples.append(
                        (shard, image_offset, image_size, label_offset, label_size)
                    )
                    self.data.append(f"{shard.name}/{key}.png")
                    self.widths.append(max(1, int(width * self.img_height / height)))

        if not self.samples:
            raise ValueError(f"Base WebDataset sem amostras em {self.dataset_dir}")

        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shard_files"] = {}
        state["_pid"] = None
        return state

    def __getitem__(self, idx):
        shard, image_offset, image_size, label_offset, label_size = self.samples[idx]
        if self._pid != os.getpid():
            self._shard_files = {}
            self._pid = os.getpid()

        shard_file = self._shard_files.get(shard)
        if shard_file is None:
            shard_file = open(shard, "rb")  # noqa: SIM115 - reused for this shard
            self._shard_files[shard] = shard_file

        shard_file.seek(image_offset)
        img_bytes = shard_file.read(image_size)
        shard_file.seek(label_offset)
        label = shard_file.read(label_size).decode("utf-8")

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

        new_w = max(1, int(w * self.img_height / h))
        img = cv2.resize(img, (int(new_w), int(self.img_height)))

        img_tensor = torch.from_numpy(img).unsqueeze(0).float().div_(255)

        # Convert text to numerical indices
        target = [CHAR2IDX[c] for c in label if c in CHAR2IDX]
        target_length = len(target)

        return img_tensor, torch.tensor(target, dtype=torch.long), target_length


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
            # nn.Dropout2d(0.15),
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

    def forward(self, x, input_lengths=None, enforce_sorted=False):
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
                enforce_sorted=enforce_sorted,
            )
            packed_out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=features.shape[1]
            )

        logits = self.fc(out)  # (B, 64, num_classes)
        return logits

    def _init_new_rows(self, start: int):
        # a fresh row must start viable: at the fine-tune lr Adam moves a weight
        # only ~0.05 per run, so a zeroed row would never learn to fire
        with torch.no_grad():
            reference_weight = self.fc.weight.data[:start]
            reference_bias = self.fc.bias.data[:start]
            new_weight = self.fc.weight.data[start:]
            new_weight.copy_(
                reference_weight.mean(dim=0) + 0.01 * torch.randn_like(new_weight)
            )
            self.fc.bias.data[start:].fill_(float(reference_bias.min()))

    def load_if_exists(self) -> bool:
        file = checkpoint_path(self.name, "ocr")
        if file.exists():
            content = torch.load(file, weights_only=True)
            state_dict = content["model"]

            # ALPHABET grew: keep the learned rows, re-init the appended ones
            fc_weight = state_dict.get("fc.weight")
            if fc_weight is not None:
                rows = min(fc_weight.shape[0], self.fc.out_features)
                with torch.no_grad():
                    self.fc.weight.data[:rows] = fc_weight[:rows]
                    self.fc.bias.data[:rows] = state_dict["fc.bias"][:rows]

                # shortcut: bias below -5 means the row was never trained (runs
                # 0.4.6/0.4.7 muted new chars at -10); re-init those too
                start = rows
                if start == self.fc.out_features:
                    muted = (self.fc.bias.data < -5).nonzero()
                    start = int(muted[0]) if len(muted) else self.fc.out_features
                if start < self.fc.out_features:
                    self._init_new_rows(start)

                state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if not key.startswith("fc.")
                }
                state_dict["fc.weight"] = self.fc.weight.data
                state_dict["fc.bias"] = self.fc.bias.data

            # Tenta carregar normalmente primeiro
            try:
                self.load_state_dict(state_dict, strict=True)
                print(f"Model {self.name} loaded successfully.")
            except RuntimeError:
                print(
                    "Checkpoint antigo detectado, ajustando chaves para nova arquitetura..."
                )
                # Se falhar, aplica a lógica de correção apenas agora
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("cnn."):
                        idx = int(k.split(".")[1])
                        if idx >= 7:  # Camadas após o dropout inserido na posição 7
                            new_k = k.replace(f"cnn.{idx}.", f"cnn.{idx + 1}.")
                            new_state_dict[new_k] = v
                        else:
                            new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=True)

            return True

        print(f"Model {self.name} doesn't exist")
        return False

    def save_model(self, output_name=""):
        name = self.name if output_name == "" else output_name
        file = checkpoint_path(name, "ocr")
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict()}, file)

        return name


image_augmentations = v2.Compose(
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


def ocr_collate_fn(batch, augment=False):
    # batch: list of tuples (img_tensor, target_tensor, target_length)
    images, targets, target_lengths = zip(*batch)
    if augment:
        images = [image_augmentations(img) for img in images]

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
