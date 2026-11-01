import unittest

from src.ocr.process import pos_processing_line
from src.search_block.analyzer import Box, Line


class PosProcessingLineTests(unittest.TestCase):
    def test_promotes_wide_letters_but_leaves_narrow_l_lowercase(self):
        line = Line(
            [
                Box(0, 0, 6, 10),
                Box(8, 0, 5, 10),
                Box(15, 0, 3, 10),
            ]
        )

        self.assertEqual(
            pos_processing_line(
                line,
                ["o", "s", "l"],
                [
                    [("o", 49.98), ("O", 28.82)],
                    [("s", 76.06), ("S", 20.07)],
                    [("l", 52.47), ("L", 34.18)],
                ],
                4,
            ),
            ["O", "S", "l"],
        )

    def test_uses_digit_candidates_only_in_numeric_tokens(self):
        line = Line(
            [
                Box(0, 0, 7, 10),
                Box(8, 0, 7, 10),
                Box(20, 0, 7, 10),
                Box(28, 0, 7, 10),
            ]
        )

        self.assertEqual(
            pos_processing_line(
                line,
                ["1", "S", "B", "O"],
                [
                    [("1", 80.0), ("I", 10.0)],
                    [("S", 70.0), ("5", 20.0)],
                    [("B", 80.0), ("8", 10.0)],
                    [("O", 80.0), ("0", 10.0)],
                ],
                4,
            ),
            ["1", "5", "B", "O"],
        )


if __name__ == "__main__":
    unittest.main()
