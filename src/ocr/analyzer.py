import statistics

import cv2
import numpy as np

from src.utils.paths import test_image_path


class Box:
    def __init__(self, x=0, y=0, w=0, h=0) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def is_zero(self):
        return self.x + self.y + self.w + self.h == 0

    def set_second_coord(self, x1, y1):
        self.w = x1 - self.x
        self.h = y1 - self.y

    def get_values(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def get_second_coord(self):
        return self.x + self.w, self.y + self.h

    def get_value_by_idx(self, idx):
        match idx:
            case 0:
                return self.x
            case 1:
                return self.y
            case 2:
                return self.w
            case 3:
                return self.h

    def set_values(self, values: tuple[int, int, int, int]):
        self.x, self.y, self.w, self.h = values

    def set_x(self, x):
        end_x = self.x + self.w
        self.x = x
        self.w = end_x - x

    def set_y(self, y):
        end_y = self.y + self.h
        self.y = y
        self.h = end_y - y

    def __str__(self) -> str:
        return f"X: [{self.x}] | Y: [{self.y}] | W: [{self.w}] | H: [{self.h}]"

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }


class Line:
    def __init__(self, boxes: list[Box] | None) -> None:
        self.bbox = Box()
        self.boxes: list[Box] = []

        if boxes:
            for box in boxes:
                self.add_box(box)

    def add_box(self, box: Box):
        self.boxes.append(box)

        if len(self.boxes) == 1:
            x, y, w, h = box.get_values()
            self.bbox.set_values((x, y, w, h))
            return

        self.bbox.set_x(min(self.bbox.x, box.x))
        self.bbox.set_y(min(self.bbox.y, box.y))

        x, y = self.bbox.get_second_coord()
        x1, y1 = box.get_second_coord()

        self.bbox.set_second_coord(max(x, x1), max(y, y1))

    def get_box_by_index(self, idx):
        return self.boxes[idx]

    def sort_boxes(self, inverse=False):
        self.boxes = sorted(self.boxes, key=lambda item: item.x)


def group_paragraphs(lines: list[Line], ratio=1.5) -> list[int]:
    """Paragraph index per line, from the page's vertical rhythm.

    The baseline is the median of the line's component bottoms: bbox edges
    move with accents and descenders, bottoms mostly sit on the baseline.
    """
    ids = [0] * len(lines)
    if len(lines) < 2:
        return ids

    order = sorted(range(len(lines)), key=lambda i: lines[i].bbox.y)
    baselines = [
        statistics.median(
            [b.y + b.h for b in lines[i].boxes] or [lines[i].bbox.y + lines[i].bbox.h]
        )
        for i in order
    ]
    advances = [baselines[j + 1] - baselines[j] for j in range(len(baselines) - 1)]

    # the pitch is the smallest common advance: paragraphs only ever widen the
    # gap, so the median lands on the paragraph gap when short ones dominate
    pitch = max(1, sorted(advances)[len(advances) // 10])

    current = 0
    for j, advance in enumerate(advances):
        if advance > ratio * pitch:
            current += 1
        ids[order[j + 1]] = current

    return ids


def segment_page(img: np.ndarray, inverted=False) -> list[Line]:
    # Otsu adapts to the ink level, so faded scans (ink lighter than 50) still segment
    _, tresh = cv2.threshold(
        img,
        0,
        255,
        (cv2.THRESH_BINARY if not inverted else cv2.THRESH_BINARY_INV)
        | cv2.THRESH_OTSU,
    )

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
            y += 1
            h -= 2
            bbox = Box(x, y, w, h)
            raw_boxes.append(bbox)

    return sort_boxes_by_line(raw_boxes)


def character_segmentation(image_name: str, inverted=False):
    # read image on gray scale
    img = cv2.imread(f"{test_image_path(image_name)}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Invalid file {image_name}.png")

    return img, segment_page(img, inverted)


def sort_boxes_by_line(boxes: list[Box], gap=2) -> list[Line]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b.y)

    lines: list[Line] = []
    curr_line = Line([boxes[0]])

    for b in boxes[1:]:
        top = curr_line.bbox.y
        bottom = curr_line.bbox.y + curr_line.bbox.h
        # same line when the vertical spans overlap; the gap covers accents
        # and dots sitting just above or below the line box
        if b.y < bottom + gap and b.y + b.h > top - gap:
            curr_line.add_box(b)
        else:
            # add the line to lines_list, sorting them by x
            curr_line.sort_boxes()
            lines.append(curr_line)
            curr_line = Line([b])

    if curr_line:
        curr_line.sort_boxes()
        lines.append(curr_line)

    # sort lines by y
    lines = sorted(lines, key=lambda l: l.get_box_by_index(0).y)

    return lines
