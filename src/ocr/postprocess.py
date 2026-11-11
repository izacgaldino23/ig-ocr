"""Lexicon post-processing for CRNN output: fixes accent and case slips
on real Portuguese words (floresta corpus). Words that exist in the corpus
are never touched, so correct rare forms like "à" stay intact."""

import re
import sys
import unicodedata
from collections import Counter

import nltk

from src.utils.paths import DATA_DIR


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


class LexiconCorrector:
    def __init__(self) -> None:
        # corpus lives where src/crnn/generator.py downloads it
        nltk.data.path.append(str(DATA_DIR / "floresta"))
        from nltk.corpus import floresta

        word_counts = Counter(
            word.casefold()
            for word in floresta.words()
            if isinstance(word, str) and word.isalpha()
        )
        self.valid_words = set(word_counts)

        # accent-stripped form -> most frequent real spelling
        best: dict[str, tuple[str, int]] = {}
        for word, freq in word_counts.items():
            key = strip_accents(word)
            if freq > best.get(key, ("", 0))[1]:
                best[key] = (word, freq)
        self.lexicon = {key: word for key, (word, _) in best.items()}

    def _correct_token(self, token: str) -> str:
        # keep punctuation attached to the word edges
        match = re.fullmatch(r"(\W*)(.*?)(\W*)", token, re.DOTALL)
        if not match:
            return token
        head, core, tail = match.groups()

        if not core or core.casefold() in self.valid_words:
            return token

        word = self.lexicon.get(strip_accents(core.casefold()))
        if word is None:
            return token

        # preserve the token's case pattern
        if core.isupper():
            word = word.upper()
        elif core[:1].isupper():
            word = word.capitalize()
        return head + word + tail

    def correct_line(self, text: str) -> str:
        return " ".join(self._correct_token(token) for token in text.split(" "))


if __name__ == "__main__":
    # Windows console may not be UTF-8 by default
    sys.stdout.reconfigure(encoding="utf-8")

    corrector = LexiconCorrector()

    assert corrector.correct_line("nao vai") == "não vai"
    assert corrector.correct_line("MEDICO") == "MÉDICO"
    assert corrector.correct_line("Medico, saida.") == "Médico, saída."
    # valid words and unknown words pass through untouched
    assert corrector.correct_line("você fica") == "você fica"
    assert corrector.correct_line("lorem explicabo") == "lorem explicabo"

    print("LexiconCorrector self-check OK")
