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
        self.baseline = 0
        self.baseline_defined = False

        if boxes:
            for box in boxes:
                self.add_box(box)

    def add_box(self, box: Box):
        self.boxes.append(box)

        if len(self.boxes) == 1:
            x, y, w, h = box.get_values()
            self.bbox.set_values((x, y, w, h))
            self.baseline = y + h
            return

        # Update baseline
        box_y = box.y + box.h
        if not self.baseline_defined and abs(box_y - self.baseline) > 2:
            self.baseline = min(self.baseline, box_y)
            self.baseline_defined = True

        self.bbox.set_x(min(self.bbox.x, box.x))
        self.bbox.set_y(min(self.bbox.y, box.y))

        x, y = self.bbox.get_second_coord()
        x1, y1 = box.get_second_coord()

        self.bbox.set_second_coord(max(x, x1), max(y, y1))

    def get_box_by_index(self, idx):
        return self.boxes[idx]

    def sort_boxes(self, inverse=False):
        self.boxes = sorted(self.boxes, key=lambda item: item.x)

    def normalize_boxes_height(self):
        for box in self.boxes:
            _, y = self.bbox.get_second_coord()
            x1, y1 = box.get_second_coord()

            box.set_y(min(self.bbox.y, box.y))

            box.set_second_coord(x1, max(y, y1))
            box.h = max(box.h, y - box.y)


def character_segmentation(image_name: str, inverted=False):
    # read image on gray scale
    img = cv2.imread(f"{test_image_path(image_name)}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Invalid file {image_name}.png")

    # simple binary conversion (for middle gray colors)
    _, tresh = cv2.threshold(
        img, 50, 255, cv2.THRESH_BINARY if not inverted else cv2.THRESH_BINARY_INV
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

    sorted_boxes_by_line = sort_boxes_by_line(raw_boxes)

    return img, sorted_boxes_by_line


def sort_boxes_by_line(boxes: list[Box], tolerance=25) -> list[Line]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b.y)

    lines: list[Line] = []
    curr_line = Line([boxes[0]])

    avg_y: int = curr_line.get_box_by_index(0).y
    for b in boxes[1:]:
        if abs(b.y - avg_y) < tolerance:  # check if both are in the same line
            curr_line.add_box(b)
            avg_y = int((avg_y + b.y) / 2)
        else:
            # add the line to lines_list, sorting them by x
            curr_line.sort_boxes()
            lines.append(curr_line)
            curr_line = Line([b])
            avg_y = b.y

    if curr_line:
        curr_line.sort_boxes()
        lines.append(curr_line)

    # sort lines by y
    lines = sorted(lines, key=lambda l: l.get_box_by_index(0).y)

    # adjust all lines by baseline
    max_line_height = lines[0].baseline - lines[0].bbox.y
    for line in lines[1:]:
        max_line_height = max(max_line_height, line.baseline - line.bbox.y)

    # normalize boxes inside each line
    # curr_y = 0
    # for line in lines:
    #     curr_y = line.bbox.y + line.bbox.h
    #     if line.baseline - line.bbox.y < max_line_height:
    #         line.bbox.set_y(line.baseline - max_line_height)
    # line.normalize_boxes_height()

    # parse the boxes to a 1d list with all boxes in the correct order
    # final_boxes = [box for line in lines for box in line]

    return lines


def boxes_to_line(lines: list[Line]) -> list[Line]:
    return []


if __name__ == "__main__":
    image_name = "3216"
    img, lines = character_segmentation(image_name, True)
    # print(lines)
    # cv2.imwrite(f"{test_image_path(image_name)}-dilatted.png", img)
    for line in lines:
        start = (line.bbox.x, line.bbox.y)
        end = line.bbox.get_second_coord()
        cv2.rectangle(img, start, end, 0, 1)
        # for b in line.boxes:
        #     start = (b.x, b.y)
        #     end = b.get_second_coord()
        #     cv2.rectangle(img, start, end, 0, 1)

        # cv2.line(
        #     img,
        #     [line.bbox.x - 4, line.bbox.y + line.bbox.h],
        #     [line.bbox.x + line.bbox.w + 4, line.bbox.y + line.bbox.h],
        #     0,
        #     2,
        # )

    h, w = img.shape

    height = 900
    width = (height * w) / h

    resized_image = cv2.resize(img, (int(width), int(height)))

    cv2.imshow("boxes", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
