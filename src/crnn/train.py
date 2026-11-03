from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.crnn.model import (
    BLANK_IDX,
    CHAR2IDX,
    CRNN,
    OCRDataset,
    ocr_collate_fn,
)
from src.utils.paths import DATA_DIR


class Train:
    def __init__(self, model_name, batch_size=32, workers=4) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(len(CHAR2IDX) + 1, model_name)
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.0001, weight_decay=1e-5
        )
        self.model_name = model_name

        dataset_dir = DATA_DIR / "words-dataset"

        self.dataset = OCRDataset(dataset_dir, train=True)
        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            collate_fn=ocr_collate_fn,
        )
        torch.cuda.synchronize(self.device)

        self.model.load_if_exists()
        self.model.to(self.device)

    def execute(self, epochs=5):
        self.model.train()
        total_batches = len(self.dataloader)

        # Simple learning rate scheduler without requiring validation
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels, target_lengths, input_lengths) in enumerate(
                self.dataloader, 1
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward Pass
                preds = self.model(images)  # Format: (B, T, Num_Classes)
                preds = preds.permute(
                    1, 0, 2
                )  # CTCLoss requires format (T, B, Num_Classes)
                log_probability = torch.nn.functional.log_softmax(preds, dim=2)

                loss = self.criterion(
                    log_probability, labels, input_lengths, target_lengths
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

                epoch_loss += loss.item()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{current_time}] Epoch [{epoch + 1}/{epochs}] | "
                    f"Batch [{batch_idx}/{total_batches}] | "
                    f"Loss: {loss.item():.4f}"
                )

            scheduler.step()
            avg_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{current_time}] === End of Epoch {epoch + 1}/{epochs} | "
                f"Average Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} ==="
            )

        self.model.save_model()
        print(f"Model saved successfully as {self.model_name}")


if __name__ == "__main__":
    train = Train("0.1", batch_size=32)
    train.execute(epochs=10)
