import unittest

from src.ocr.analyzer import Box, Line, group_paragraphs
from src.ocr.recognize import page_text


def line_on(baseline: int, left: int = 100, height: int = 20) -> Line:
    """A line whose components sit on `baseline`."""
    return Line(
        [
            Box(left, baseline - height, 8, height),
            Box(left + 12, baseline - height, 8, height),
            Box(left + 24, baseline - height, 8, height),
        ]
    )


def lines_on(baselines: list[int]) -> list[Line]:
    return [line_on(baseline) for baseline in baselines]


class GroupParagraphsTests(unittest.TestCase):
    def test_splits_on_the_wider_gap(self) -> None:
        lines = lines_on([0, 60, 120, 240, 300])
        self.assertEqual(group_paragraphs(lines), [0, 0, 0, 1, 1])

    def test_short_paragraphs_do_not_hide_the_pitch(self) -> None:
        # real advances from page 31144: paragraphs are as common as lines, so
        # a median pitch would land on the paragraph gap and find no break
        lines = lines_on([0, 118, 236, 354, 472, 531, 649, 767])
        self.assertEqual(group_paragraphs(lines), [0, 1, 2, 3, 4, 4, 5, 6])

    def test_single_line_and_empty(self) -> None:
        self.assertEqual(group_paragraphs([]), [])
        self.assertEqual(group_paragraphs(lines_on([0])), [0])

    def test_ignores_a_small_advance_outlier(self) -> None:
        lines = lines_on([0, 58, 118, 238, 298])
        self.assertEqual(group_paragraphs(lines), [0, 0, 0, 1, 1])

    def test_ids_follow_the_input_order(self) -> None:
        ordered = lines_on([0, 60, 180])
        self.assertEqual(
            group_paragraphs([ordered[2], ordered[0], ordered[1]]), [1, 0, 0]
        )

    def test_ratio_tightens_the_break(self) -> None:
        lines = lines_on([0, 60, 150])
        self.assertEqual(group_paragraphs(lines, ratio=3), [0, 0, 0])


class PageTextTests(unittest.TestCase):
    def test_visual_keeps_line_breaks(self) -> None:
        self.assertEqual(page_text(["a", "b", "c"], [0, 0, 1]), "a\nb\n\nc")

    def test_flow_joins_wrapped_lines(self) -> None:
        self.assertEqual(page_text(["a", "b", "c"], [0, 0, 1], flow=True), "a b\n\nc")

    def test_flow_keeps_hyphenated_words_together(self) -> None:
        # the corpus token is "bastar-se"; the wrap landed on the hyphen
        self.assertEqual(
            page_text(["conseguem bastar-", "se agora"], [0, 0], flow=True),
            "conseguem bastar-se agora",
        )

    def test_visual_keeps_the_hyphen_at_the_line_end(self) -> None:
        self.assertEqual(
            page_text(["conseguem bastar-", "se agora"], [0, 0]),
            "conseguem bastar-\nse agora",
        )


if __name__ == "__main__":
    unittest.main()
