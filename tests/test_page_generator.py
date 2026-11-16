import unittest

from src.datagen.page_generator import PageGenerator


class RenderBoundsTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.generator = PageGenerator()
        except (FileNotFoundError, OSError, ValueError) as error:
            self.skipTest(f"font catalog unavailable: {error}")

    def test_no_line_bbox_leaves_the_page(self) -> None:
        # a bbox past the page edge comes back padded black when cropped, which
        # pairs a black block with the full label and ruins the sample
        sentence = "palavra " * 120
        for _ in range(5):
            image, lines, _ = self.generator.render([sentence])
            self.assertTrue(lines)
            for line in lines:
                x, y, w, h = line.bbox.get_values()
                self.assertLessEqual(x + w, image.width, line.text)
                self.assertLessEqual(y + h, image.height, line.text)

    def test_no_word_is_dropped_at_a_line_break(self) -> None:
        # the word that overflows must start the next line, not vanish
        sentence = " ".join(f"p{index}" for index in range(40))
        _, lines, _ = self.generator.render([sentence])
        self.assertEqual(" ".join(line.text for line in lines), sentence)


if __name__ == "__main__":
    unittest.main()
