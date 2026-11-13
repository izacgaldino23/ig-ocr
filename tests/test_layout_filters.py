import unittest

import cv2
import numpy as np

from src.ocr.analyzer import (
    Box,
    Line,
    drop_junk_lines,
    find_page,
    segment_page,
)


def bar(mask: np.ndarray, x: int, y: int, w: int, h: int) -> Line:
    """Paint a solid bar into the mask and return it as a line."""
    mask[y : y + h, x : x + w] = 255
    return Line([Box(x, y, w, h)])


def normal_lines(mask: np.ndarray) -> list[Line]:
    """Three ordinary text lines, so the median height is not the case under test."""
    return [bar(mask, 50, y, 300, 20) for y in (100, 130, 160)]


class SegmentPageTests(unittest.TestCase):
    def test_a_closed_frame_does_not_swallow_the_text_inside(self) -> None:
        # the frame is one contour; with RETR_EXTERNAL the text inside its hole
        # disappears with it and the whole page comes back as a single line
        img = np.full((300, 400), 255, np.uint8)
        cv2.rectangle(img, (10, 10), (389, 289), 0, 6)
        for y in (80, 150, 220):
            cv2.rectangle(img, (60, y), (340, y + 18), 0, -1)

        lines = segment_page(img, inverted=True)

        self.assertEqual(len(lines), 3)
        self.assertLess(max(line.bbox.h for line in lines), img.shape[0] * 0.5)

    def test_reports_what_it_discarded(self) -> None:
        img = np.full((300, 400), 255, np.uint8)
        cv2.rectangle(img, (10, 10), (389, 289), 0, 6)
        cv2.rectangle(img, (60, 80), (340, 98), 0, -1)

        dropped: list = []
        segment_page(img, inverted=True, dropped=dropped)

        self.assertEqual([reason for reason, _ in dropped], ["frame"])


class JunkLineTests(unittest.TestCase):
    def test_thin_rule_is_dropped(self) -> None:
        mask = np.zeros((200, 400), np.uint8)
        lines = [bar(mask, 50, 20, 300, 20) for _ in range(3)]
        rule = bar(mask, 50, 150, 300, 3)

        kept = drop_junk_lines(lines + [rule], mask)

        self.assertEqual(len(kept), 3)
        self.assertNotIn(rule, kept)

    def test_faint_small_line_survives(self) -> None:
        # a faded scan: 0.8% ink, but normal line size, so it is still text
        mask = np.zeros((200, 400), np.uint8)
        lines = [bar(mask, 50, 20, 300, 20) for _ in range(3)]
        faint = Line([Box(50, 150, 100, 20)])
        mask[150:170:5, 50:150:25] = 255

        kept = drop_junk_lines(lines + [faint], mask)

        self.assertIn(faint, kept)

    def test_empty_large_box_is_dropped(self) -> None:
        mask = np.zeros((200, 400), np.uint8)
        lines = [bar(mask, 50, 20, 300, 20) for _ in range(3)]
        empty = Line([Box(0, 150, 380, 45)])
        mask[150:195:6, 0:380:40] = 255

        kept = drop_junk_lines(lines + [empty], mask)

        self.assertNotIn(empty, kept)

    def test_splits_two_rows_fused_into_one_line(self) -> None:
        mask = np.zeros((200, 400), np.uint8)
        bar(mask, 50, 20, 300, 20)
        bar(mask, 50, 60, 300, 20)
        fused = Line([Box(50, 20, 300, 20), Box(50, 60, 300, 20)])

        kept = drop_junk_lines([fused] + normal_lines(mask), mask)

        self.assertEqual(len(kept), 5)
        self.assertEqual(sorted(line.bbox.y for line in kept), [20, 60, 100, 130, 160])

    def test_splits_a_row_bridged_by_a_centred_value(self) -> None:
        # a receipt total in a bigger font sits between two rows and bridges
        # them: there is no valley across the box, so the cut comes from the
        # gutter beside that value
        mask = np.zeros((200, 400), np.uint8)
        bar(mask, 50, 20, 200, 20)
        bar(mask, 50, 60, 200, 20)
        mask[20:80, 300:400] = 255  # the tall value bridging both rows
        fused = Line(
            [Box(50, 20, 200, 20), Box(50, 60, 200, 20), Box(300, 20, 100, 60)]
        )

        kept = drop_junk_lines([fused] + normal_lines(mask), mask)

        self.assertEqual(len(kept), 6)
        self.assertEqual(
            sorted((line.bbox.x, line.bbox.y) for line in kept),
            [(50, 20), (50, 60), (50, 100), (50, 130), (50, 160), (300, 20)],
        )

    def test_a_word_space_is_not_a_cell_gutter(self) -> None:
        # 10px between words on a 20px line: splitting there would shred a
        # value into pieces, so it must stay one line
        mask = np.zeros((200, 400), np.uint8)
        bar(mask, 50, 20, 240, 20)
        bar(mask, 50, 60, 240, 20)
        mask[20:80, 300:400] = 255  # the tall value, 10px from the words
        fused = Line(
            [Box(50, 20, 240, 20), Box(50, 60, 240, 20), Box(300, 20, 100, 60)]
        )

        kept = drop_junk_lines([fused] + normal_lines(mask), mask)

        self.assertIn(fused, kept)


class FindPageTests(unittest.TestCase):
    def test_a_clean_scan_is_left_alone(self) -> None:
        img = np.full((400, 300), 255, np.uint8)
        cv2.rectangle(img, (40, 40), (260, 60), 0, -1)

        self.assertIsNone(find_page(img))

    def test_straightens_a_page_over_a_background(self) -> None:
        photo = np.full((600, 800), 60, np.uint8)  # dark desk
        cv2.fillPoly(
            photo,
            [np.array([[200, 120], [600, 100], [620, 480], [180, 500]], np.int32)],
            235,
        )
        cv2.rectangle(photo, (250, 200), (550, 220), 30, -1)

        page = find_page(photo)

        self.assertIsNotNone(page)
        height, width = page.shape[:2]
        self.assertAlmostEqual(width, 440, delta=44)
        self.assertAlmostEqual(height, 380, delta=38)
        self.assertGreater(page.mean(), 180)  # the desk is gone


if __name__ == "__main__":
    unittest.main()
