import argparse
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Sampler, random_split

from src.crnn.model import (
    BLANK_IDX,
    CHAR2IDX,
    CRNN,
    IDX2CHAR,
    OCRDataset,
    ocr_collate_fn,
)
from src.utils.paths import DATA_DIR, PROJECT_ROOT


class WidthBucketBatchSampler(Sampler):
    def __init__(self, dataset, widths, batch_size, bucket_batches=100):
        self.dataset = dataset
        self.widths = widths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_batches

    def __iter__(self):
        indices = torch.randperm(len(self.dataset)).tolist()
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.widths.__getitem__, reverse=True)
            batches.extend(
                bucket[index : index + self.batch_size]
                for index in range(0, len(bucket), self.batch_size)
            )

        for index in torch.randperm(len(batches)).tolist():
            yield batches[index]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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
        training_widths = [
            self.dataset.widths[index] for index in training_data.indices
        ]
        self.dataloader = DataLoader(
            training_data,
            batch_sampler=WidthBucketBatchSampler(
                training_data, training_widths, batch_size
            ),
            pin_memory=self.device.type == "cuda",
            num_workers=workers,
            persistent_workers=workers > 0,
            collate_fn=partial(ocr_collate_fn, augment=True),
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
        loss_sum = torch.zeros((), device=self.device)
        with torch.inference_mode():
            for (
                images,
                labels,
                target_lengths,
                input_lengths,
                _,
            ) in self.validation_dataloader:
                images = images.to(self.device, non_blocking=True)
                labels_cpu = labels
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images, input_lengths)
                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
                loss_sum += loss * images.size(0)

                predictions = logits.argmax(dim=2).cpu()
                target_lengths_list = target_lengths.tolist()
                input_lengths_list = input_lengths.tolist()
                start = 0
                for i, target_length in enumerate(target_lengths_list):
                    target = labels_cpu[start : start + target_length].tolist()
                    start += target_length
                    previous = BLANK_IDX
                    prediction = []
                    for index in predictions[i, : input_lengths_list[i]].tolist():
                        if index != BLANK_IDX and index != previous:
                            prediction.append(IDX2CHAR.get(index, ""))
                        previous = index
                    correct += prediction == [IDX2CHAR[index] for index in target]
                    total += 1

        return (
            correct / total if total else 0.0,
            loss_sum.item() / total if total else 0.0,
        )

    def execute(self, epochs=10, profile=False):
        if profile:
            epochs = 1

        profile_dir = PROJECT_ROOT / "artifacts" / "profiler" / "crnn"
        if profile:
            print(f"TensorBoard profiler traces (10 batches per epoch): {profile_dir}")

        total_batches = len(self.dataloader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        best_accuracy = -1.0
        best_validation_loss = float("inf")
        best_weights = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = torch.zeros((), device=self.device)
            profiler = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        *(
                            [torch.profiler.ProfilerActivity.CUDA]
                            if self.device.type == "cuda"
                            else []
                        ),
                    ],
                    schedule=torch.profiler.schedule(
                        wait=2, warmup=2, active=10, repeat=1
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir)
                    ),
                    record_shapes=True,
                    profile_memory=True,
                )
                if profile
                else nullcontext()
            )
            with profiler:
                for batch_idx, (
                    images,
                    labels,
                    target_lengths,
                    input_lengths,
                    image_widths,
                ) in enumerate(self.dataloader, 1):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Forward Pass
                    batch_logits = self.model(
                        images, input_lengths, enforce_sorted=True
                    )

                    log_probability = batch_logits.permute(
                        1, 0, 2
                    )  # CTCLoss requires format (T, B, Num_Classes)
                    log_probability = torch.nn.functional.log_softmax(
                        log_probability, dim=2
                    )

                    loss = self.criterion(
                        log_probability, labels, input_lengths, target_lengths
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.detach()
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if (
                        batch_idx == 1
                        or batch_idx % 25 == 0
                        or batch_idx == total_batches
                    ):
                        print(
                            f"[{current_time}] Epoch [{epoch + 1}/{epochs}] | "
                            f"Batch [{batch_idx}/{total_batches}]"
                        )

                    if profile and profiler:
                        profiler.step()

            validation_accuracy, validation_loss = self._validate()
            is_best = validation_accuracy > best_accuracy or (
                validation_accuracy == best_accuracy
                and validation_loss < best_validation_loss
            )
            if is_best and not profile:
                best_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                best_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.model.save_model(self.output_name)

            scheduler.step()
            avg_loss = epoch_loss.item() / total_batches if total_batches > 0 else 0.0
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{current_time}] === End of Epoch {epoch + 1}/{epochs} | "
                f"Average Loss: {avg_loss:.4f} | "
                f"Val Accuracy: {validation_accuracy:.2%} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
                f"{' | Best checkpoint' if is_best and not profile else ''} ==="
            )

        if profile:
            return
        if best_weights is None:
            raise RuntimeError("Training completed without a validation checkpoint")
        self.model.load_state_dict(best_weights)
        print(
            f"Best model saved as {self.output_name or self.model_name} "
            f"with validation accuracy {best_accuracy:.2%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run one epoch and save a TensorBoard profiler trace.",
    )
    args = parser.parse_args()

    train = Train("0.4", batch_size=64)
    train.execute(epochs=1 if args.profile else 25, profile=args.profile)
