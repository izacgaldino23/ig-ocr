from datetime import datetime

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.crnn.model import (
    BLANK_IDX,
    CHAR2IDX,
    CRNN,
    IDX2CHAR,
    OCRDataset,
    gpu_augmentations,
    ocr_collate_fn,
)
from src.utils.paths import DATA_DIR


class Train:
    def __init__(
        self,
        model_name,
        output_name="",
        batch_size=32,
        workers=4,
        learning_rate=1e-5,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(len(CHAR2IDX) + 1, model_name)
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.model_name = model_name
        self.output_name = output_name

        dataset_dir = DATA_DIR / "words-dataset"
        self.dataset = OCRDataset(dataset_dir, train=True)
        if len(self.dataset) < 2:
            raise ValueError(
                "At least two training samples are required for validation"
            )

        validation_size = max(1, round(len(self.dataset) * 0.1))
        training_size = len(self.dataset) - validation_size
        training_data, validation_data = random_split(
            self.dataset,
            (training_size, validation_size),
            generator=torch.Generator().manual_seed(42),
        )
        self.dataloader = DataLoader(
            training_data,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            collate_fn=ocr_collate_fn,
        )
        self.validation_dataloader = DataLoader(
            validation_data,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            collate_fn=ocr_collate_fn,
        )
        torch.cuda.synchronize(self.device)

        self.model.load_if_exists()
        self.model.to(self.device)

    def _validate(self):
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.inference_mode():
            for (
                images,
                labels,
                target_lengths,
                input_lengths,
                _,
            ) in self.validation_dataloader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images, input_lengths)
                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
                loss_sum += loss.item() * images.size(0)

                start = 0
                for i, target_length in enumerate(target_lengths.tolist()):
                    target = labels[start : start + target_length].tolist()
                    start += target_length
                    previous = BLANK_IDX
                    prediction = []
                    for index in logits[i, : input_lengths[i]].argmax(dim=1).tolist():
                        if index != BLANK_IDX and index != previous:
                            prediction.append(IDX2CHAR.get(index, ""))
                        previous = index
                    correct += prediction == [IDX2CHAR[index] for index in target]
                    total += 1

        return correct / total if total else 0.0, loss_sum / total if total else 0.0

    def execute(self, epochs=10):
        total_batches = len(self.dataloader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        best_accuracy = -1.0
        best_validation_loss = float("inf")
        best_weights = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, (
                images,
                labels,
                target_lengths,
                input_lengths,
                image_widths,
            ) in enumerate(self.dataloader, 1):
                cpu_images_pinned = images.is_pinned()
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                augmented_images = []
                for image, width in zip(images, image_widths.tolist()):
                    image = gpu_augmentations(image[:, :, :width])
                    augmented_images.append(
                        F.pad(
                            image,
                            (0, images.shape[-1] - width, 0, 0),
                            mode="replicate",
                        )
                    )
                images = torch.stack(augmented_images)

                # Forward Pass
                batch_logits = self.model(images, input_lengths)
                log_probability = batch_logits.permute(
                    1, 0, 2
                )  # CTCLoss requires format (T, B, Num_Classes)
                log_probability = torch.nn.functional.log_softmax(
                    log_probability, dim=2
                )

                loss = self.criterion(
                    log_probability, labels, input_lengths, target_lengths
                )

                # debug_batch = batch_idx == total_batches
                debug_batch = False
                if epoch == 0 and batch_idx == 1:
                    print(
                        f"Device Debug | model={next(self.model.parameters()).device} "
                        f"images={images.device} labels={labels.device} "
                        f"logits={batch_logits.device} "
                        f"target_lengths={target_lengths.device} "
                        f"input_lengths={input_lengths.device} "
                        f"image_widths={image_widths.device} "
                        f"cpu_batch_pinned={cpu_images_pinned}"
                    )
                if debug_batch:
                    valid_steps = (
                        torch.arange(batch_logits.shape[1], device=self.device)[None, :]
                        < input_lengths.to(self.device)[:, None]
                    )
                    pred_indices = batch_logits.argmax(dim=2)
                    blank_ratio = (
                        (pred_indices[valid_steps] == BLANK_IDX).float().mean().item()
                    )
                    blank_probability = (
                        batch_logits[valid_steps]
                        .softmax(dim=-1)[:, BLANK_IDX]
                        .mean()
                        .item()
                    )
                    first_target_len = target_lengths[0].item()
                    first_target = "".join(
                        IDX2CHAR[idx]
                        for idx in labels[:first_target_len].tolist()
                        if idx in IDX2CHAR
                    )
                    first_path = []
                    previous = BLANK_IDX
                    for idx in pred_indices[0, : input_lengths[0]].tolist():
                        if idx != BLANK_IDX and idx != previous and idx in IDX2CHAR:
                            first_path.append(IDX2CHAR[idx])
                        previous = idx
                    print(
                        f"CTC Debug | Epoch {epoch + 1} Batch {batch_idx} | "
                        f"image_shape={tuple(images.shape)} logits={tuple(batch_logits.shape)} "
                        f"input_lengths={input_lengths.min().item()}..{input_lengths.max().item()} "
                        f"target_lengths={target_lengths.min().item()}..{target_lengths.max().item()} | "
                        f"blank_argmax={blank_ratio:.3f} "
                        f"mean_blank_prob={blank_probability:.3f} "
                        f"first_target={first_target!r} "
                        f"first_pred={''.join(first_path)!r}"
                    )

                self.optimizer.zero_grad()
                loss.backward()
                if debug_batch:
                    grad_norm = self.model.fc.weight.grad.norm().item()
                    weight_before_step = self.model.fc.weight.detach().clone()
                self.optimizer.step()
                if debug_batch:
                    update_norm = (
                        (self.model.fc.weight.detach() - weight_before_step)
                        .norm()
                        .item()
                    )
                    print(
                        f"CTC Debug | loss={loss.item():.4f} "
                        f"fc_grad_norm={grad_norm:.6g} "
                        f"fc_update_norm={update_norm:.6g}"
                    )
                epoch_loss += loss.item()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if batch_idx == 1 or batch_idx == total_batches:
                    print(
                        f"[{current_time}] Epoch [{epoch + 1}/{epochs}] | "
                        f"Batch [{batch_idx}/{total_batches}] | "
                        f"Loss: {loss.item():.4f}"
                    )

            validation_accuracy, validation_loss = self._validate()
            is_best = validation_accuracy > best_accuracy or (
                validation_accuracy == best_accuracy
                and validation_loss < best_validation_loss
            )
            if is_best:
                best_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                best_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.model.save_model(self.output_name)

            scheduler.step()
            avg_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{current_time}] === End of Epoch {epoch + 1}/{epochs} | "
                f"Average Loss: {avg_loss:.4f} | "
                f"Val Accuracy: {validation_accuracy:.2%} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
                f"{' | Best checkpoint' if is_best else ''} ==="
            )

        if best_weights is None:
            raise RuntimeError("Training completed without a validation checkpoint")
        self.model.load_state_dict(best_weights)
        print(
            f"Best model saved as {self.output_name or self.model_name} "
            f"with validation accuracy {best_accuracy:.2%}"
        )


if __name__ == "__main__":
    train = Train("0.2", output_name="0.3", batch_size=32)
    train.execute(epochs=100)
