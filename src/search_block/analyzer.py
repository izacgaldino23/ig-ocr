from typing import TypeAlias

import cv2
import numpy as np

from src.char_reader_nn.paths import test_image_path

Box: TypeAlias = list[int]
Line: TypeAlias = list[Box]


def character_segmentation(image_name: str):
    # read image on gray scale
    img = cv2.imread(f"{test_image_path(image_name)}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Invalid file {image_name}.png")

    # simple binaryzation (for middle gray colors)
    _, tresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # dilate to join closer white from the same character
    kernel = np.ones((5, 1), np.uint8)
    img_dilated = cv2.dilate(tresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes: list[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 3 or h >= 3:  # filter small noises
            bbox = [x, y, w, h]
            raw_boxes.append(bbox)

    # print(f"{len(raw_boxes)} raw boxes found")
    sorted_boxes = sort_boxes_by_line(raw_boxes)

    return img_dilated, sorted_boxes


def sort_boxes_by_line(boxes: list[Box], tollerance=25):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])

    lines: list[Line] = []
    curr_line: Line = [boxes[0]]

    count = 0
    avg_y: int = curr_line[0][1]
    for b in boxes[1:]:
        if abs(b[1] - avg_y) < tollerance:  # check if both are in the same line
            curr_line.append(b)
            avg_y = int((avg_y + b[1]) / 2)
        else:
            # add the line to lines_list, sorting them by x
            lines.append(sorted(curr_line, key=lambda item: item[0]))
            curr_line = [b]
            avg_y = b[1]

        count += 1

    # print(f"looped over {count + 1} boxes")
    if curr_line:
        lines.append(sorted(curr_line, key=lambda item: item[0]))

    # print(f"{len(lines)} different lines found")

    # sorte lines by y
    lines = sorted(lines, key=lambda l: l[0][1])

    # parse the boxes to a 1d list with all boxes in the correct order
    final_boxes = [box for line in lines for box in line]

    # print(f"There are {len(final_boxes)} final boxes")

    return final_boxes


if __name__ == "__main__":
    image_name = "9857"
    img, boxes = character_segmentation(image_name)
    # cv2.imwrite(f"{test_image_path(image_name)}-dilatted.png", img)
    # for b in boxes:
    #     start = (b[0], b[1])
    #     end = (b[0] + b[2], b[1] + b[3])
    #     cv2.rectangle(img, start, end, (250, 0, 0), 1)

    # cv2.imshow("boxes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
