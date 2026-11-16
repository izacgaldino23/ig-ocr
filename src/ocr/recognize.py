import argparse
import json
import re
import sys

import cv2
import numpy as np
import torch

from src.ocr.analyzer import Line, character_segmentation, group_paragraphs
from src.ocr.model import CHAR2IDX, CRNN
from src.ocr.postprocess import LexiconCorrector
from src.training.evaluate import count_wrong_chars, decode_ctc
from src.utils.paths import test_image_path


# process_line receives a cut of the original image of a whole text line
# resizes it proportionally to 48px height (training default) and
# returns a [1, 1, 48, W] tensor ready for the CRNN model
def process_line(roi: np.ndarray, img_height=48) -> torch.Tensor:
    # 1px white border so the ink never touches the tensor edges
    roi = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    h, w = roi.shape

    # proportional resize keeping the ratio, same as OCRDataset
    new_w = max(1, int(w * (img_height / h)))
    resized = cv2.resize(roi, (new_w, img_height))

    tensor = torch.from_numpy(resized).unsqueeze(0).float().div_(255)

    # add batch dimension: [1, 1, 48, W]
    return tensor.unsqueeze(0)


def line_crop(img: np.ndarray, line: Line) -> np.ndarray:
    x, y, w, h = line.bbox.get_values()
    if w <= 0 or h <= 0:
        return img[0:0, 0:0]

    # trailing period blobs are discarded by the analyzer noise filter,
    # so the crop must reach past the line bbox to include them
    pad = int(h * 0.3)
    x_end = min(x + w + pad, img.shape[1])

    return img[y : y + h, x:x_end]


def recognize_line(
    model: CRNN, device: torch.device, img: np.ndarray, line: Line
) -> str:
    roi = line_crop(img, line)
    if roi.size == 0:
        return ""

    input_tensor = process_line(roi).to(device)

    with torch.no_grad():
        # Forward pass: (1, T, Num_Classes)
        preds = model(input_tensor)

    return decode_ctc(preds[0])


def process_page(
    image_name: str, model: CRNN, device: torch.device, dropped: list | None = None
) -> list[tuple[Line, str]]:
    # pages are rendered black text on white background
    img, lines = character_segmentation(image_name, inverted=True, dropped=dropped)

    return [(line, recognize_line(model, device, img, line)) for line in lines]


def _join_flow(lines: list[str], lexicon: set[str] | None = None) -> str:
    """Wrapped lines joined by a space, except across a hyphen.

    Two different things end a line with a hyphen: a real compound (bastar-se)
    and a word broken by hyphenation (docu- / mento). The lexicon decides: the
    hyphen is dropped only when the joined form is a known word.

    Without a lexicon both forms are unknown, so the hyphen is kept - correct
    for the corpus, wrong for a hyphenated scan.
    """
    text = ""
    for line in lines:
        head, tail = text.rstrip(), line.lstrip()
        if not text:
            text = line
        elif not head.endswith("-"):
            text = head + " " + tail
        else:
            stem = head[:-1]
            prefix = stem[max(stem.rfind(" "), stem.rfind("\n")) + 1 :]
            suffix = re.match(r"[^\W\d_]+", tail)
            joined = prefix + (suffix.group() if suffix else "")
            if lexicon and len(joined) > len(prefix) and joined.casefold() in lexicon:
                text = head[:-1] + tail
            else:
                text = head + tail
    return text


def fix_hyphen_spacing(text: str) -> str:
    """Close the space inside a hyphenated word ("bastar- se" -> "bastar-se").

    A dash used as punctuation is spaced on both sides ("a - b"), so only a
    hyphen glued to the previous letter is treated as part of the word.
    """
    return re.sub(r"(?<=\w)- (?=\w)", "-", text)


def page_text(
    texts: list[str],
    paragraph_ids: list[int],
    flow=False,
    lexicon: set[str] | None = None,
) -> str:
    """Paragraphs separated by a blank line.

    flow joins the wrapped lines of a paragraph with a space instead of keeping
    the visual line breaks.
    """
    blocks: dict[int, list[str]] = {}
    for text, paragraph_id in zip(texts, paragraph_ids):
        blocks.setdefault(paragraph_id, []).append(fix_hyphen_spacing(text))

    if flow:
        return "\n\n".join(_join_flow(block, lexicon) for block in blocks.values())
    return "\n\n".join("\n".join(block) for block in blocks.values())


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
    parser.add_argument("image", help="image name inside data/test_images")
    parser.add_argument(
        "checkpoint", help="checkpoint name inside artifacts/checkpoints/ocr"
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="enable the lexicon accent/case corrector (real Portuguese text)",
    )
    parser.add_argument(
        "--para-ratio",
        type=float,
        default=1.5,
        help="line advance over the page pitch that starts a new paragraph",
    )
    parser.add_argument(
        "--flow",
        action="store_true",
        help="join a paragraph's wrapped lines with a space instead of a newline",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print every box the segmenter discarded and why",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(len(CHAR2IDX) + 1, args.checkpoint)
    if not model.load_if_exists():
        raise FileNotFoundError(f"Model {args.checkpoint} does not exist")
    model = model.to(device)

    # Disable gradient calculation.
    model.eval()

    dropped: list = [] if args.debug else None
    results = process_page(args.image, model, device, dropped)
    paragraphs = group_paragraphs([line for line, _ in results], args.para_ratio)

    # the lexicon resolves hyphenation in flow text, so both flags need it
    corrector = LexiconCorrector() if (args.postprocess or args.flow) else None
    lexicon = corrector.valid_words if corrector else None
    predicted_text = page_text(
        [text for _, text in results], paragraphs, args.flow, lexicon
    )

    print(f"Image: {args.image}")
    print(f"Predicted text:\n{predicted_text}\n")

    if dropped is not None:
        print(f"Discarded boxes: {len(dropped)}")
        for reason, box in dropped:
            print(f"  {reason:6} {box}")
        print()

    has_gt = test_image_path(f"{args.image}.json").exists()
    if args.postprocess:
        corrected = [(line, corrector.correct_line(text)) for line, text in results]
        corrected_text = page_text(
            [text for _, text in corrected], paragraphs, args.flow, lexicon
        )
        print(f"Corrected text: \n{corrected_text}\n")
        if has_gt:
            compare_with_json(args.image, results, show_lines=False, title="RAW")
            compare_with_json(args.image, corrected, title="CORRECTED")
    elif has_gt:
        compare_with_json(args.image, results)

    if not has_gt:
        print("(no ground-truth json next to the image, comparison skipped)")
