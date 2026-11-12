import argparse
import json
import sys

import cv2
import numpy as np
import torch

from src.crnn.evaluate import count_wrong_chars, decode_ctc
from src.crnn.model import CHAR2IDX, CRNN
from src.ocr.postprocess import LexiconCorrector
from src.search_block.analyzer import Line, character_segmentation
from src.utils.paths import test_image_path


# process_line receives a cut of the original image of a whole text line
# resizes it proportionally to 32px height (training default) and
# returns a [1, 1, 32, W] tensor ready for the CRNN model
def process_line(roi: np.ndarray, img_height=32) -> torch.Tensor:
    # 1px white border so the ink never touches the tensor edges
    roi = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = roi.shape

    # proportional resize keeping the ratio, same as OCRDataset
    new_w = max(1, int(w * (img_height / h)))
    resized = cv2.resize(roi, (new_w, img_height))

    tensor = torch.from_numpy(resized).unsqueeze(0).float().div_(255)

    # add batch dimension: [1, 1, 32, W]
    return tensor.unsqueeze(0)


def recognize_line(
    model: CRNN, device: torch.device, img: np.ndarray, line: Line
) -> str:
    x, y, w, h = line.bbox.get_values()
    if w <= 0 or h <= 0:
        return ""

    # trailing period blobs are discarded by the analyzer noise filter,
    # so the crop must reach past the line bbox to include them
    pad = int(h * 0.3)
    x_end = min(x + w + pad, img.shape[1])
    input_tensor = process_line(img[y : y + h, x:x_end]).to(device)

    with torch.no_grad():
        # Forward pass: (1, T, Num_Classes)
        preds = model(input_tensor)

    return decode_ctc(preds[0])


def process_page(
    image_name: str, model: CRNN, device: torch.device
) -> list[tuple[Line, str]]:
    # pages are rendered black text on white background
    img, lines = character_segmentation(image_name, inverted=True)

    return [(line, recognize_line(model, device, img, line)) for line in lines]


def compare_with_json(
    image_name: str,
    results: list[tuple[Line, str]],
    show_lines=True,
    title="RESULTS",
):
    # ground truth written by PageGenerator.generate() next to the image
    with open(test_image_path(f"{image_name}.json"), encoding="utf-8") as json_file:
        clues = json.load(json_file)["clues"]

    totals = {"replace": 0, "delete": 0, "insert": 0}
    total_chars = 0

    for line, pred_text in results:
        # match the clue with the largest vertical overlap with the detected line
        clue = max(
            clues,
            key=lambda c: (
                min(line.bbox.y + line.bbox.h, c["bbox"]["y"] + c["bbox"]["h"])
                - max(line.bbox.y, c["bbox"]["y"])
            ),
        )

        gt_text = clue["text"]
        mistakes = count_wrong_chars(pred_text, gt_text)

        total_chars += len(gt_text)
        for key in totals:
            totals[key] += mistakes[key]

        if show_lines:
            print(f"exp:  {gt_text}")
            print(f"got:  {pred_text}")
            print(f"distance: {mistakes['distance']}\n")

    distance = sum(totals.values())
    print(f"== {title} ==")
    for key, value in totals.items():
        print(f"{key} \t{value}")
    if total_chars:
        correct = total_chars - distance
        print(f"char accuracy: {correct / total_chars:.2%} ({correct}/{total_chars})")
    return total_chars, distance


if __name__ == "__main__":
    # Windows console may not be UTF-8 by default
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    parser = argparse.ArgumentParser(description="CRNN page OCR")
    parser.add_argument(
        "image", nargs="?", default="48576", help="image inside data/test_images"
    )
    parser.add_argument("checkpoint", nargs="?", default="0.4.3-ft")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="enable the lexicon accent/case corrector (real Portuguese text)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(len(CHAR2IDX) + 1, args.checkpoint)
    if not model.load_if_exists():
        raise FileNotFoundError(f"Model {args.checkpoint} does not exist")
    model = model.to(device)

    # Disable gradient calculation.
    model.eval()

    results = process_page(args.image, model, device)
    predicted_text = "\n".join(text for _, text in results)

    print(f"Image: {args.image}")
    print(f"Predicted text:\n{predicted_text}\n")

    has_gt = test_image_path(f"{args.image}.json").exists()
    if args.postprocess:
        corrector = LexiconCorrector()
        corrected = [(line, corrector.correct_line(text)) for line, text in results]
        corrected_text = "\n".join(text for _, text in corrected)
        print(f"Corrected text: \n{corrected_text}\n")
        if has_gt:
            compare_with_json(args.image, results, show_lines=False, title="RAW")
            compare_with_json(args.image, corrected, title="CORRECTED")
    elif has_gt:
        compare_with_json(args.image, results)

    if not has_gt:
        print("(no ground-truth json next to the image, comparison skipped)")
