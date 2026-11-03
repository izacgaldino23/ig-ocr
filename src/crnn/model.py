import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

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
        self.data = []
        self.targets = []
        self.dataset_dir = dataset_dir / ("train" if train else "test")
        with open(self.dataset_dir / "labels.txt", "r", encoding="utf-8") as f:
            for line in f:
                (file_path, label) = line.strip().split("\t")
                self.data.append(file_path)
                self.targets.append(label)

        self.img_height = img_height

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        full_path = str(self.dataset_dir / img_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None or img.size == 0:
            raise FileNotFoundError(f"Error reading file {full_path} (label [{label}])")

        # Proportional img resize keeping the ratio
        h, w = img.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions ({h}x{w}) for {full_path}")

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

        # 3. Output Projection
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # (B, 256, 1, W) W = img_width / 4
        features = features.squeeze(2)  # (B, 256, W)
        features = features.permute(0, 2, 1)  # (B, W, 256) -> (Batch, Time, Feat)

        out, _ = self.rnn(features)  # (B, W, 256)
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

    def save_model(self):
        file = checkpoint_path(self.name, "ocr")
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict()}, file)


def ocr_collate_fn(batch):
    # batch: list of tuples (img_tensor, target_tensor, target_length)
    images, targets, target_lengths = zip(*batch)
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

    return images_tensor, targets_tensor, torch.tensor(target_lengths), input_lengths
