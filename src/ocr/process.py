import os

import cv2
import numpy as np
import torch
from cv2.typing import MatLike

from src.neural_network.config import init_config
from src.neural_network.model import MEAN, STD, model_from_checkpoint
from src.search_block.analyzer import Box, Line, character_segmentation
from src.utils.paths import DEBUG_FOLDER, checkpoint_path


# process_roi receives a cut of the original image of each character
# stabilize it and return a 28x28 tensor
# ready for EMINIST Neural Network processing
# ROI means Region Of Interest
def process_roi(
    roi: np.ndarray, rotate: bool = False, printout=False
) -> tuple[MatLike, torch.Tensor]:
    if rotate:
        roi = cv2.transpose(roi)

    # _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    h, w = roi.shape

    # get the bigger side
    max_side = max(h, w)

    # add a padding rounding the characther
    padding = int(max_side * 0.2)
    square_size = max_side + 2 * padding

    # create a black square image
    square_img = np.zeros((square_size, square_size), dtype=np.uint8)

    # get central point to paste the cutted character
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2
    square_img[y_offset : y_offset + h, x_offset : x_offset + w] = roi

    # resize to 28x28, EMINIST default size
    resized_24x24 = cv2.resize(square_img, (24, 24), interpolation=cv2.INTER_AREA)
    padded_28x28 = np.zeros((28, 28), dtype=np.uint8)
    padded_28x28[2:26, 2:26] = resized_24x24
    resized = padded_28x28

    # convert to tensor pytorch and normalize (0 to 1)
    tensor_img = torch.from_numpy(resized).float() / 255.0

    # add chanel and batch dimensions: [1, 1, 28, 28]
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)

    # identical normalization with the training
    tensor_img = (tensor_img - MEAN) / STD

    return resized, tensor_img


def remove_debug_images():
    files = os.listdir(DEBUG_FOLDER)
    for f in files:
        if os.path.isfile(f"{DEBUG_FOLDER}/{f}"):
            os.remove(f"{DEBUG_FOLDER}/{f}")


def pos_processing_letter(line: Line, box: Box, char: str):
    threshold = line.bbox.h * 0.05
    if abs(box.y - line.bbox.y) > threshold and char.isupper():
        return char.lower()

    return char


def box_features(line: Line, box: Box) -> dict[str, float]:
    line_height = max(line.baseline - line.bbox.y, 1)
    return {
        "height_ratio": box.h / line_height,
        "width_height_ratio": round(box.w / max(box.h, 1), 2),
        "top_offset_ratio": (box.y - line.bbox.y) / line_height,
        "baseline_offset_ratio": (line.baseline - (box.y + box.h)) / line_height,
    }


if __name__ == "__main__":
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    remove_debug_images()

    images = [
        [
            "9800",
            "Lorem Ipsum is simply dummy text of the printing and typesetting industry",
        ],
        ["6771", "Fifi found five fluffy foxes beside the little mill"],
        ["9515", "I like Lily and Mimi who use 10 old maps while Oliver and Sam read"],
    ]
    current_index = 0
    image_name = images[current_index][0]
    original_text = images[current_index][1]

    img, lines = character_segmentation(image_name)
    predicted_text = ""

    args, device = init_config(True)

    file = checkpoint_path(args.input)
    content = torch.load(file)
    class_names: list[str] = content["classes"]
    model = model_from_checkpoint(content).to(device)

    # Disable gradient calculation.
    model.eval()
    tollerance = 4
    new_line = False

    results = {"correct": 0, "case_swap": 0, "number_swap": 0, "wrong_char": 0}
    similar = "fFlIi0Oo"

    for line in lines:
        curr_x = -1
        for box in line.boxes:
            x, y, w, h = box.get_values()
            roi = img[y : y + h, x : x + w]
            resized, input_tensor = process_roi(roi)

            if new_line:
                new_line = False
                predicted_text += " "

            if curr_x != -1 and abs(x - curr_x) > tollerance:  # has space
                # print(f"Space size {abs(x - curr_x)}")
                predicted_text += " "

            curr_x = x + w
            curr_y = y

            with torch.no_grad():
                output = model(input_tensor)

                logits = output[0]
                probs = torch.softmax(logits, dim=0)

                _, pred = torch.max(output, 1)
                model_char = class_names[int(pred.item())]
                char = model_char

                char = pos_processing_letter(line, box, char)
                predicted_text += char
                idx = len(predicted_text) - 1
                expected_char = original_text[idx]

                if char == expected_char:
                    results["correct"] += 1
                elif char.lower() == expected_char.lower():
                    results["case_swap"] += 1
                elif char.isalnum() != expected_char.isalnum():
                    results["number_swap"] += 1
                else:
                    results["wrong_char"] += 1

                top_candidates = [
                    (class_names[i], round(float(probs[i] * 100), 2))
                    for i in probs.topk(min(5, probs.numel())).indices
                ]

                if char != expected_char:
                    # Save the 28x28 ROI image for a misclassified character.
                    name = f"{idx + 1}_expe_{expected_char}_got_{char}"
                    print(
                        name,
                        f"model={model_char}",
                        # f"geometry={box_features(line, box)}",
                        top_candidates,
                    )
                    filename = f"{DEBUG_FOLDER}/{name}.png"
                    cv2.imwrite(filename, resized)
                elif model_char in similar:
                    print(
                        "SIMILAR",
                        f"model={model_char}",
                        f"pospr={char}",
                        # f"geometry={box_features(line, box)}",
                        top_candidates,
                    )

        new_line = True

    print(f"Original text:  \t '{original_text}'")
    print(f"Predicted text: \t '{predicted_text}'")
    print("== RESULTS ==")
    for k, v in results.items():
        print(f"{k} \t{v}")
