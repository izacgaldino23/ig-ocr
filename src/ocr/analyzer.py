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


# a text line is a few percent of the page height; a frame or page sized art is
# most of it (measured: 98.3% for a framed diploma vs 1.1-4.6% for real lines)
FRAME_HEIGHT = 0.15
# a rule or dotted separator is far shorter than real text (measured 3-4px vs 22px)
THIN_RATIO = 0.4
# a box with almost no ink is not text (measured 1.7% vs 13-44% for real lines)
EMPTY_DENSITY = 0.03
# a fused pair of lines is about twice the page's median height (measured 42px vs 17px)
FUSED_RATIO = 1.6
# a gutter between two cells is far wider than a word space (measured 57px vs 10px
# on a 17px receipt line); a sliver on either side is not a cell
GUTTER_RATIO = 1.0
GUTTER_SIDE = 0.25
# a page that fills the frame is not a photo: nothing to straighten
PAGE_AREA = (0.2, 0.98)
PAGE_ASPECT = (0.4, 2.5)


def segment_page(
    img: np.ndarray, inverted=False, dropped: list | None = None
) -> list[Line]:
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

    # RETR_CCOMP keeps the text inside a closed frame: the frame is a top level
    # contour, its interior a hole, and what sits inside that hole is top level
    # again. RETR_EXTERNAL discards it together with the frame.
    contours, hierarchy = cv2.findContours(
        img_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    page_h = img.shape[0]
    raw_boxes: list[Box] = []
    for idx, c in enumerate(contours):
        if hierarchy is not None and hierarchy[0][idx][3] != -1:
            continue  # a hole: letter counters, frame interior

        x, y, w, h = cv2.boundingRect(c)
        if h > page_h * FRAME_HEIGHT:  # a frame or page sized art, never a line
            _report(dropped, "frame", Box(x, y, w, h))
            continue
        if w >= 3 or h >= 3:  # filter small noises
            y += 1
            h -= 2
            raw_boxes.append(Box(x, y, w, h))

    return drop_junk_lines(sort_boxes_by_line(raw_boxes), tresh, dropped)


def _report(dropped: list | None, reason: str, box: Box) -> None:
    if dropped is not None:
        dropped.append((reason, box))


def _ink_density(mask: np.ndarray, box: Box) -> float:
    roi = mask[box.y : box.y + box.h, box.x : box.x + box.w]
    return float(np.mean(roi == 255)) if roi.size else 0.0


def drop_junk_lines(
    lines: list[Line], mask: np.ndarray, dropped: list | None = None
) -> list[Line]:
    """Drop boxes that are not text and split a line that is really two."""
    if not lines:
        return lines

    median_h = statistics.median([line.bbox.h for line in lines]) or 1
    page_w = mask.shape[1]

    kept: list[Line] = []
    for line in lines:
        bbox = line.bbox
        if bbox.h < median_h * THIN_RATIO:
            _report(dropped, "thin", bbox)
        elif _ink_density(mask, bbox) < EMPTY_DENSITY and (
            bbox.w > page_w * 0.5 or bbox.h > median_h * 2
        ):
            # sparse only means junk when the box is also big: faint small text
            # on a faded scan has to survive
            _report(dropped, "empty", bbox)
        else:
            kept.extend(_split_fused(line, mask, median_h))
    return kept


def _split_fused(line: Line, mask: np.ndarray, median_h: float) -> list[Line]:
    """Split a line that is really two rows, at an ink valley or a cell gutter.

    A value in a bigger font can sit centred between two rows and bridge them
    (a receipt total between DINHEIRO and TROCO): there is no valley across the
    whole box, so the cut has to come from the gutter beside that value.
    """
    if line.bbox.h <= median_h * FUSED_RATIO:
        return [line]
    parts = _cut_box(line, mask, median_h, 0)
    return parts or [line]


def _cut_box(
    line: Line, mask: np.ndarray, median_h: float, depth: int
) -> list[Line] | None:
    """Recursive cuts of a fused box, or None when no cut is safe."""
    if depth >= 3 or len(line.boxes) < 2:
        return None
    if line.bbox.h <= median_h * FUSED_RATIO:
        return None

    sides = None
    cut = _valley_row(mask, line.bbox)
    if cut is not None:
        sides = (
            [box for box in line.boxes if box.y + box.h / 2 < cut],
            [box for box in line.boxes if box.y + box.h / 2 >= cut],
        )
    else:
        gutter = _gutter_col(mask, line.bbox, median_h)
        if gutter is not None:
            sides = (
                [box for box in line.boxes if box.x + box.w / 2 < gutter],
                [box for box in line.boxes if box.x + box.w / 2 >= gutter],
            )

    if sides is None or not sides[0] or not sides[1]:
        return None

    parts: list[Line] = []
    for group in sides:
        part = Line(group)
        parts.extend(_cut_box(part, mask, median_h, depth + 1) or [part])
    return sorted(parts, key=lambda p: (p.bbox.y, p.bbox.x))


def _empty_run(values: np.ndarray) -> tuple[int, int]:
    """Longest run of true values as (start, length), ignoring the edges."""
    best, best_start, run, start = 0, 0, 0, 0
    for idx, value in enumerate(values):
        if value:
            if run == 0:
                start = idx
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return best_start, best


def _gutter_col(mask: np.ndarray, box: Box, median_h: float) -> int | None:
    """Column that separates two cells of the same row, or None."""
    cols = (mask[box.y : box.y + box.h, box.x : box.x + box.w] == 255).sum(axis=0)
    if cols.size < 4 or cols.max() == 0:
        return None

    start, width = _empty_run(cols == 0)
    if width < median_h * GUTTER_RATIO or start == 0 or start + width >= cols.size:
        return None
    # a word gap is not a cell gutter, and neither side may be a sliver
    if start < box.w * GUTTER_SIDE or (box.w - start - width) < box.w * GUTTER_SIDE:
        return None
    return box.x + start + width // 2


def _valley_row(mask: np.ndarray, box: Box, floor=0.08) -> int | None:
    """Row that separates two fused lines, or None when the ink never drops."""
    rows = mask[box.y : box.y + box.h, box.x : box.x + box.w] == 255
    counts = rows.sum(axis=1)
    if counts.size < 4 or counts.max() == 0:
        return None

    start, width = _empty_run(counts <= counts.max() * floor)

    # a valley touching an edge is the empty space above or below the line
    if width < 2 or start == 0 or start + width >= counts.size:
        return None
    return box.y + start + width // 2


def _order_corners(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), np.float32)
    total = points.sum(axis=1)
    spread = points[:, 0] - points[:, 1]
    ordered[0] = points[np.argmin(total)]  # top left
    ordered[1] = points[np.argmax(spread)]  # top right
    ordered[2] = points[np.argmax(total)]  # bottom right
    ordered[3] = points[np.argmin(spread)]  # bottom left
    return ordered


def _page_quad(contour, gray: np.ndarray, frame_area: int) -> np.ndarray | None:
    """The contour as 4 ordered corners, or None when it fails the page gate."""
    perimeter = cv2.arcLength(contour, True)
    quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return None

    if not PAGE_AREA[0] <= cv2.contourArea(quad) / frame_area <= PAGE_AREA[1]:
        return None

    _, _, w, h = cv2.boundingRect(quad)
    if not PAGE_ASPECT[0] <= w / max(h, 1) <= PAGE_ASPECT[1]:
        return None

    # the paper is lighter than whatever surrounds it
    inside = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(inside, [quad], -1, 255, -1)
    if cv2.mean(gray, inside)[0] < cv2.mean(gray, cv2.bitwise_not(inside))[0] + 10:
        return None

    return _order_corners(quad.reshape(4, 2).astype(np.float32))


def _warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(
        max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
    )
    if width < 10 or height < 10:
        return None

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_CUBIC)


def find_page(img: np.ndarray) -> np.ndarray | None:
    """Straighten the paper out of a photo, or None when the image is a page.

    The gate matters more than the detector: a wrong quadrilateral silently
    destroys the page, so anything doubtful returns None and the caller keeps
    the original image.
    """
    height, width = img.shape[:2]
    factor = min(1.0, 1000 / max(height, width))
    small = (
        cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        if factor < 1
        else img
    )

    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(contour) / frame_area < PAGE_AREA[0]:
            break  # every remaining contour is smaller
        quad = _page_quad(contour, blur, frame_area)
        if quad is None:
            continue
        warped = _warp(img, quad / factor)
        if warped is not None:
            return warped
    return None


def character_segmentation(
    image_name: str, inverted=False, dropped: list | None = None
):
    # read image on gray scale
    img = cv2.imread(f"{test_image_path(image_name)}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Invalid file {image_name}.png")

    page = find_page(img)
    if page is not None:
        img = page

    return img, segment_page(img, inverted, dropped)


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
