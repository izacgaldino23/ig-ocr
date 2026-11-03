import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.crnn.model import (
    BLANK_IDX,
    CHAR2IDX,
    CRNN,
    IDX2CHAR,
    OCRDataset,
    ocr_collate_fn,
)
from src.utils.paths import DATA_DIR


class Train:
    def __init__(self, model_name, batch_size=32, workers=4) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(len(CHAR2IDX) + 1, model_name).to(self.device)
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        self.model_name = model_name

        dataset_dir = DATA_DIR / "words-dataset"

        self.dataset = OCRDataset(dataset_dir)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            collate_fn=ocr_collate_fn,
        )

        self.model.load_if_exists()

    def execute(self):
        # Simplified training (1 example epoch)
        self.model.train()
        for images, labels, target_lengths in self.dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward Pass
            preds = self.model(images)  # Format: (B, T, Num_Classes)
            preds = preds.permute(
                1, 0, 2
            )  # CTCLoss requires format (T, B, Num_Classes)
            log_probability = torch.nn.functional.log_softmax(preds, dim=2)

            # Time step dimensions for the batch
            input_lengths = torch.tensor(
                [img.shape[2] // 4 for img in images], dtype=torch.long
            )

            loss = self.criterion(
                log_probability, labels, input_lengths, target_lengths
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

        self.model.save_model()

    def decode_ctc(self, preds):
        # predictions: tensors in the form (T, Num_Classes)
        pred_indices = torch.argmax(preds, dim=-1).tolist()

        decoded_text = []
        last_idx = BLANK_IDX

        for idx in pred_indices:
            if idx != BLANK_IDX and idx != last_idx:
                decoded_text.append(IDX2CHAR[idx])
            last_idx = idx

        return "".join(decoded_text)


if __name__ == "__main__":
    train = Train("0.1")
    train.execute()
