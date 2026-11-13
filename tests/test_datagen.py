import unittest

from src.datagen.page_generator import rejoin_hyphenated


class RejoinHyphenatedTests(unittest.TestCase):
    def test_joins_the_treebank_split_word(self) -> None:
        # floresta yields "bastar-" + "se" for the single token "bastar-se"
        self.assertEqual(
            rejoin_hyphenated(["conseguem", "bastar-", "se", "em"]),
            ["conseguem", "bastar-se", "em"],
        )

    def test_keeps_a_standalone_dash_apart(self) -> None:
        self.assertEqual(
            rejoin_hyphenated(["Benfica", "-", "Sporting"]),
            ["Benfica", "-", "Sporting"],
        )

    def test_leaves_whole_words_alone(self) -> None:
        self.assertEqual(
            rejoin_hyphenated(["norte-americano", "e", "guarda-redes"]),
            ["norte-americano", "e", "guarda-redes"],
        )


if __name__ == "__main__":
    unittest.main()
