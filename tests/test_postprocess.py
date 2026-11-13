import unittest

from src.ocr.postprocess import LexiconCorrector, strip_accents


class LexiconCorrectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corrector = LexiconCorrector()

    def test_fixes_missing_accents(self) -> None:
        self.assertEqual(self.corrector.correct_line("nao vai"), "não vai")

    def test_preserves_case_and_punctuation(self) -> None:
        self.assertEqual(self.corrector.correct_line("MEDICO"), "MÉDICO")
        self.assertEqual(
            self.corrector.correct_line("Medico, saida."), "Médico, saída."
        )

    def test_leaves_valid_and_unknown_words_untouched(self) -> None:
        self.assertEqual(self.corrector.correct_line("você fica"), "você fica")
        self.assertEqual(
            self.corrector.correct_line("lorem explicabo"), "lorem explicabo"
        )

    def test_keeps_contractions_the_corpus_omits(self) -> None:
        self.assertEqual(
            self.corrector.correct_line("da casa do pai"), "da casa do pai"
        )

    def test_mixed_case_follows_the_letter_majority(self) -> None:
        # the model's case slips are per letter, so the majority is the intent
        self.assertEqual(self.corrector.correct_line("SUiÇOS"), "SUÍÇOS")


class StripAccentsTests(unittest.TestCase):
    def test_strips_combining_marks(self) -> None:
        self.assertEqual(strip_accents("Médico"), "Medico")


if __name__ == "__main__":
    unittest.main()
