import json
import os

from src.search_block.analyzer import character_segmentation
from src.search_block.image_generator import generate_test_image
from src.utils.paths import TEST_IMAGE_DIR, test_image_path

texts = [
    "i j",
    "a1 i2 j3",
    "E F H K X Z",
    "A B C D ",
    "a b c d e f g h",
    "A1a B2b C3c",
    "Line1\nLine2\nLine3",
    "12345\ni j k l\n67890",
]


def remove_images(file_names: list[str]):
    for f in file_names:
        os.remove(test_image_path(f"{f}.png"))
        os.remove(test_image_path(f"{f}.json"))


def main():
    file_names = []

    # clean test images folder
    if TEST_IMAGE_DIR.exists():
        files = os.listdir(TEST_IMAGE_DIR)
        for f in files:
            if os.path.isfile(TEST_IMAGE_DIR / f):
                os.remove(TEST_IMAGE_DIR / f)

    # generate test images
    for t in texts:
        file_names.append(generate_test_image(t))

    # run character_segmentation to get the boxes
    correct = []
    labels = ["x", "y", "w", "h"]
    avg_diff = [0, 0, 0, 0]
    tollerance = 25
    try:
        for f in file_names:
            _, lines = character_segmentation(f)

            # Get the reference coordinates from the file.
            json_path = test_image_path(f"{f}.json")

            errors = []
            # Load the JSON data.
            with open(json_path) as json_file:
                data = json.load(json_file)

                boxes_clue: list = data["box"]

                for line in lines:
                    boxes = line.boxes
                    if len(boxes_clue) != len(boxes):
                        print(
                            f"Image [{f}] has a different number of boxes. JSON: [{len(boxes_clue)}]. Prediction: [{len(boxes)}]"
                        )
                    for i, box in enumerate(boxes):
                        clue = boxes_clue[i]
                        tollerances = [
                            tollerance,
                            tollerance,
                            tollerance,
                            tollerance,
                        ]

                        for idx in range(4):
                            v1 = box.get_value_by_idx(idx)
                            v2 = clue["bbox"][idx]
                            diff: int = abs(v1 - v2)
                            if diff > tollerances[idx]:
                                errors.append(
                                    f"image [{f}], letter [{clue['char']}] idx [{i + 1}], [{labels[idx]}] value, white [{clue['bbox']}]\t red [{box}]"
                                )
                                if avg_diff[idx] == 0:
                                    avg_diff[idx] = diff
                                else:
                                    avg_diff[idx] = int((avg_diff[idx] + diff) / 2)

                    if not errors:
                        print(f"Image {f} is OK")
                        correct.append(f)
                    else:
                        print(f"Image {f} errors:")
                        for e in errors:
                            print(f"\t => {e}")

    finally:
        remove_images(correct)
        for i in range(len(avg_diff)):
            print(f"Average difference for {labels[i]} = {avg_diff[i]}")


if __name__ == "__main__":
    main()
