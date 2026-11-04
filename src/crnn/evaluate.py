import csv

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


def count_wrong_chars(str1, str2):
    distances = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    for i in range(len(str1) + 1):
        distances[i][0] = i
    for j in range(len(str2) + 1):
        distances[0][j] = j

    for i, char1 in enumerate(str1, 1):
        for j, char2 in enumerate(str2, 1):
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + (char1 != char2),
            )

    errors = {"distance": distances[-1][-1], "replace": 0, "delete": 0, "insert": 0}
    i, j = len(str1), len(str2)
    while i or j:
        if (
            i
            and j
            and distances[i][j]
            == distances[i - 1][j - 1] + (str1[i - 1] != str2[j - 1])
        ):
            if str1[i - 1] != str2[j - 1]:
                errors["replace"] += 1
            i -= 1
            j -= 1
        elif i and distances[i][j] == distances[i - 1][j] + 1:
            errors["delete"] += 1
            i -= 1
        else:
            errors["insert"] += 1
            j -= 1

    return errors


if __name__ == "__main__":
    model_name = "0.3"
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
    headers = [
        "image_path",
        "got",
        "expected",
        "distance",
        "replace",
        "delete",
        "insert",
    ]
    sample_idx = 0
    with torch.inference_mode():
        for images, labels, target_lengths, input_lengths, _ in test_loader:
            images = images.to(device, non_blocking=True)

            # Forward Pass: retorna (B, T, Num_Classes)
            preds = model(images, input_lengths)

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
                    mistakes = count_wrong_chars(pred_text, gt_text)
                    image_path = (
                        dataset.dataset_dir / dataset.data[sample_idx]
                    ).relative_to(DATA_DIR)
                    errors.append(
                        [
                            image_path.as_posix(),
                            pred_text,
                            gt_text,
                            mistakes["distance"],
                            mistakes["replace"],
                            mistakes["delete"],
                            mistakes["insert"],
                        ]
                    )
                total += 1
                sample_idx += 1

    errors = sorted(errors, key=lambda item: item[3], reverse=True)

    errors_file = PROJECT_ROOT / "artifacts" / "evaluate_errors.csv"
    with open(errors_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(headers)
        writer.writerows(errors)

    accuracy = (correct_count / total) * 100 if total > 0 else 0.0
    print(
        f"Test dataset accuracy (Exact Match): {accuracy:.2f}% ({correct_count}/{total})"
    )
