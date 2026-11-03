import torch
from torch.utils.data import DataLoader

from src.crnn.model import (
    BLANK_IDX,
    CHAR2IDX,
    CRNN,
    IDX2CHAR,
    OCRDataset,
    ocr_collate_fn,
)
from src.utils.paths import DATA_DIR, PROJECT_ROOT


def decode_ctc(preds):
    # predictions: tensors in the form (T, Num_Classes)
    pred_indices = torch.argmax(preds, dim=-1).tolist()

    decoded_text = []
    last_idx = BLANK_IDX

    for idx in pred_indices:
        if idx != BLANK_IDX and idx != last_idx and idx in IDX2CHAR:
            decoded_text.append(IDX2CHAR[idx])
        last_idx = idx

    return "".join(decoded_text)


if __name__ == "__main__":
    model_name = "0.1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(len(CHAR2IDX) + 1, model_name).to(device)
    if not model.load_if_exists():
        raise FileNotFoundError("Model does not exist")

    # Aponta diretamente para o arquivo de labels de teste
    dataset_dir = DATA_DIR / "words-dataset"
    dataset = OCRDataset(dataset_dir, False)
    test_loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=device.type == "cuda",
        batch_size=32,
        num_workers=2,
        persistent_workers=False,
        collate_fn=ocr_collate_fn,
    )

    model.eval()
    correct_count = 0
    total = 0

    errors = []

    with torch.inference_mode():
        for images, labels, target_lengths, input_lengths in test_loader:
            images = images.to(device, non_blocking=True)

            # Forward Pass: retorna (B, T, Num_Classes)
            preds = model(images)

            start_idx = 0
            for i, target_len in enumerate(target_lengths):
                # Extrai o texto real (Ground Truth) da amostra i
                len_val = target_len.item()
                target_indices = labels[start_idx : start_idx + len_val].tolist()
                start_idx += len_val

                gt_text = "".join(
                    [IDX2CHAR[idx] for idx in target_indices if idx in IDX2CHAR]
                )

                # Decodifica a predição da amostra i
                pred_text = decode_ctc(preds[i, : input_lengths[i]])

                if pred_text == gt_text:
                    correct_count += 1
                else:
                    errors.append(f"got [{pred_text}] expected: [{gt_text}]\n")
                total += 1

            torch.cuda.empty_cache()

    errors_file = PROJECT_ROOT / "evaluate_errors.txt"
    with open(errors_file, "w", encoding="utf-8") as f:
        f.writelines(errors)

    accuracy = (correct_count / total) * 100 if total > 0 else 0.0
    print(
        f"Test dataset accuracy (Exact Match): {accuracy:.2f}% ({correct_count}/{total})"
    )
