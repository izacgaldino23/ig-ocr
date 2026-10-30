import time
from contextlib import nullcontext
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.neural_network.config import init_config
from src.neural_network.model import (
    COMPACT_CNN_ARCHITECTURE,
    COMPACT_CNN_CHANNELS,
    DEFAULT_CNN_CHANNELS,
    MEAN,
    SIMPLE_CNN_ARCHITECTURE,
    STD,
    SimpleCNN,
    fix_emnist_orientation,
    get_chars74k_dataset,
    get_emnist_dataset,
    get_gpu_augmentations,
    load_checkpoint_weights,
)
from src.utils.paths import checkpoint_path


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


def _prepare_training_data(args, device):
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
        _get_dataset_targets(train_dataset), args.seed
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
        if _get_dataset_data(train_dataset).dtype != torch.uint8:
            raise TypeError(
                f"Expected raw EMNIST images as uint8, got {_get_dataset_data(train_dataset).dtype}"
            )
        preload_start = time.perf_counter()
        training_images = (
            fix_emnist_orientation(
                _get_dataset_data(train_dataset).to(device, non_blocking=True)
            )
            .unsqueeze(1)
            .contiguous()
        )
        training_labels = _get_dataset_targets(train_dataset).to(
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


def _create_model(args, train_dataset, device):
    channels = COMPACT_CNN_CHANNELS if args.compact else DEFAULT_CNN_CHANNELS
    architecture = COMPACT_CNN_ARCHITECTURE if args.compact else SIMPLE_CNN_ARCHITECTURE
    model = SimpleCNN(
        num_classes=len(_get_dataset_classes(train_dataset)), channels=channels
    )
    file = checkpoint_path(args.input)
    print("4. Network created successfully!")

    if file.exists():
        content = torch.load(file)
        if load_checkpoint_weights(
            model,
            content["model"],
            content["classes"],
            _get_dataset_classes(train_dataset),
        ):
            print(
                "5. Compatible weights loaded; new classes will be trained from scratch."
            )
        else:
            print("5. Incompatible architecture; starting a new model.")
            if args.output == 0 or args.output == args.input:
                raise ValueError(
                    "The input checkpoint is incompatible with the ByClass model. "
                    "Choose a different output to preserve the existing checkpoint."
                )
    else:
        print("5. New model!")

    return model.to(device), architecture


def _create_loss_and_optimizer(model, train_targets, device, is_fine_tuning):
    targets = train_targets.detach().clone()
    class_counts = torch.bincount(targets).float()
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = torch.clamp(class_weights, max=100.0)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5 if is_fine_tuning else 0.001, weight_decay=1e-4
    )
    return loss_fn, optimizer


def _train_epoch(
    args,
    model,
    train_loader,
    training_images,
    training_labels,
    train_indices,
    batch_size,
    loss_fn,
    optimizer,
    gpu_transforms,
    device,
    num_batches,
    epoch,
    epochs,
):
    epoch_start = time.perf_counter()
    running_loss_sum = torch.zeros((), device=device)
    running_loss_weight = torch.zeros((), device=device)
    interval_total = 0
    correct = torch.zeros((), device=device, dtype=torch.long)
    epoch_loss_sum = torch.zeros((), device=device)
    epoch_loss_weight = torch.zeros((), device=device)
    epoch_total = 0
    epoch_correct = torch.zeros((), device=device, dtype=torch.long)

    prof_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        if args.profile and epoch == 1
        else nullcontext()
    )

    with prof_context as prof:
        if args.gpu_uint8:
            if (
                training_images is None
                or training_labels is None
                or train_indices is None
            ):
                raise RuntimeError("GPU uint8 dataset was not initialized")
            indices = train_indices[torch.randperm(len(train_indices), device=device)]
            epoch_batches = (
                (
                    training_images[indices[start : start + batch_size]],
                    training_labels[indices[start : start + batch_size]],
                )
                for start in range(0, len(train_indices), batch_size)
            )
        else:
            if train_loader is None:
                raise RuntimeError("DataLoader was not initialized")
            epoch_batches = iter(train_loader)

        for batch_idx, (images, labels) in enumerate(epoch_batches):
            if args.gpu_uint8:
                images = images.to(torch.float32).mul_(1.0 / 255.0)
                images.sub_(MEAN).div_(STD)
            else:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            if model.training:
                images = gpu_transforms(images)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_loss_weight = loss_fn.weight[labels].sum()
            batch_loss_sum = loss.detach() * batch_loss_weight
            running_loss_sum += batch_loss_sum
            running_loss_weight += batch_loss_weight
            interval_total += labels.size(0)
            correct += (outputs.detach().argmax(dim=1) == labels).sum()
            epoch_loss_sum += batch_loss_sum
            epoch_loss_weight += batch_loss_weight
            epoch_total += labels.size(0)
            epoch_correct += (outputs.detach().argmax(dim=1) == labels).sum()

            if batch_idx % 200 == 199 or batch_idx == num_batches - 1:
                avg_loss = running_loss_sum.item() / running_loss_weight.item()
                accuracy = correct.item() / interval_total
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"\t=> Progress [{epoch + 1}/{epochs}], "
                    f"Batch [{batch_idx + 1}/{num_batches}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy * 100:.2f}%"
                )
                running_loss_sum.zero_()
                running_loss_weight.zero_()
                correct.zero_()
                interval_total = 0

            if prof is not None:
                prof.step()

    epoch_duration = time.perf_counter() - epoch_start
    epoch_loss = (epoch_loss_sum / epoch_loss_weight).item()
    epoch_accuracy = epoch_correct.item() / epoch_total
    return epoch_loss, epoch_accuracy, epoch_duration


def _validate(
    model,
    validation_loader,
    training_images,
    training_labels,
    validation_indices,
    batch_size,
    loss_fn,
    class_names,
    device,
):
    model.eval()
    confusion = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.long, device=device
    )
    loss_sum = torch.zeros((), device=device)
    loss_weight = torch.zeros((), device=device)
    with torch.inference_mode():
        if validation_indices is not None:
            if training_images is None or training_labels is None:
                raise RuntimeError("GPU uint8 validation data was not initialized")
            batches = (
                (
                    training_images[validation_indices[start : start + batch_size]]
                    .to(torch.float32)
                    .mul_(1.0 / 255.0)
                    .sub_(MEAN)
                    .div_(STD),
                    training_labels[validation_indices[start : start + batch_size]],
                )
                for start in range(0, len(validation_indices), batch_size)
            )
        else:
            if validation_loader is None:
                raise RuntimeError("Validation DataLoader was not initialized")
            batches = validation_loader

        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            batch_loss = loss_fn(outputs, labels)
            batch_weight = loss_fn.weight[labels].sum()
            loss_sum += batch_loss * batch_weight
            loss_weight += batch_weight
            predicted = outputs.argmax(dim=1)
            confusion += torch.bincount(
                labels * len(class_names) + predicted,
                minlength=len(class_names) ** 2,
            ).reshape(len(class_names), len(class_names))

    confusion = confusion.cpu()
    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diagonal()
    class_accuracy = class_correct.float() / class_totals.clamp_min(1)
    accuracy = class_correct.sum().item() / class_totals.sum().item()
    macro_accuracy = class_accuracy.mean().item()
    validation_loss = (loss_sum / loss_weight).item()
    print(
        f"Validation: Loss: {validation_loss:.4f}, "
        f"Accuracy: {accuracy * 100:.2f}%, "
        f"Macro-accuracy: {macro_accuracy * 100:.2f}%"
    )
    # print("Validation accuracy by class:")
    # for index in class_accuracy.argsort().tolist():
    #     print(
    #         f"  {class_names[index]}: {class_accuracy[index]:.4f} "
    #         f"[{class_correct[index]} of {class_totals[index]}]"
    #     )
    # ======
    # off_diagonal = confusion.clone()
    # off_diagonal.fill_diagonal_(0)
    # top_confusions = torch.topk(off_diagonal.flatten(), k=min(7, off_diagonal.numel()))
    # confusions = [
    #     (
    #         class_names[index // len(class_names)],
    #         class_names[index % len(class_names)],
    #         int(count),
    #     )
    #     for index, count in zip(
    #         top_confusions.indices.tolist(), top_confusions.values.tolist()
    #     )
    #     if count > 0
    # ]
    # if confusions:
    #     print("Most common validation confusions (actual -> predicted):")
    #     for actual, predicted, count in confusions:
    #         print(f"  {actual} -> {predicted}: {count}")
    return validation_loss, accuracy, macro_accuracy


def _train(
    args,
    model,
    train_dataset,
    train_loader,
    training_images,
    training_labels,
    train_indices,
    validation_loader,
    validation_indices,
    batch_size,
    num_batches,
    loss_fn,
    optimizer,
    device,
):
    epochs = 25
    print("6. Starting training...")
    start_total = time.perf_counter()
    gpu_transforms = get_gpu_augmentations().to(device)

    for epoch in range(epochs):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] -> Epoch {epoch + 1} started")
        epoch_start = time.perf_counter()
        train_loss, train_accuracy, training_duration = _train_epoch(
            args,
            model,
            train_loader,
            training_images,
            training_labels,
            train_indices,
            batch_size,
            loss_fn,
            optimizer,
            gpu_transforms,
            device,
            num_batches,
            epoch,
            epochs,
        )
        validation_loss, validation_accuracy, macro_accuracy = _validate(
            model,
            validation_loader,
            training_images,
            training_labels,
            validation_indices,
            batch_size,
            loss_fn,
            _get_dataset_classes(train_dataset),
            device,
        )
        model.train()
        epoch_duration = time.perf_counter() - epoch_start
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] -> Epoch {epoch + 1} "
            f"finished in {epoch_duration:.2f}s"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] summary: "
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_accuracy * 100:.2f}%, "
            f"Validation loss: {validation_loss:.4f}, "
            f"Validation accuracy: {validation_accuracy * 100:.2f}%, "
            f"Macro-accuracy: {macro_accuracy * 100:.2f}%, "
            f"Training: {training_duration:.2f}s, "
            f"Total: {epoch_duration:.2f}s"
        )

    total_duration = time.perf_counter() - start_total
    print(f"7. Training completed in {total_duration:.2f}s!")


def _save_checkpoint(model, classes, architecture, args):
    file = checkpoint_path(args.output if args.output != 0 else args.input)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": classes,
            "architecture": architecture,
        },
        file,
    )
    print(f"8. Model saved successfully to {file}!")


def _get_dataset_targets(combined_dataset):
    return torch.cat([d.targets for d in combined_dataset.datasets])


def _get_dataset_data(combined_dataset):
    return torch.cat([d.data for d in combined_dataset.datasets])


def _get_dataset_classes(combined_dataset):
    return combined_dataset.datasets[0].classes


def main():
    torch.backends.cudnn.benchmark = True
    args, device = init_config()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    (
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
    ) = _prepare_training_data(args, device)
    model, architecture = _create_model(args, train_dataset, device)
    loss_fn, optimizer = _create_loss_and_optimizer(
        model,
        _get_dataset_targets(train_dataset)[train_indices],
        device,
        args.fine_tune,
    )
    _train(
        args,
        model,
        train_dataset,
        train_loader,
        training_images,
        training_labels,
        train_indices_device,
        validation_loader,
        validation_indices_device,
        batch_size,
        num_batches,
        loss_fn,
        optimizer,
        device,
    )
    _save_checkpoint(model, _get_dataset_classes(train_dataset), architecture, args)


if __name__ == "__main__":
    main()
