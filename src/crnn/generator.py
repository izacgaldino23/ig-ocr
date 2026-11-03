import glob
import os
import random

import nltk
from nltk.corpus import floresta
from PIL import ImageFont
from trdg.generators import GeneratorFromStrings

from src.utils.paths import DATA_DIR


def custom_getsize(self, text):
    bbox = self.getbbox(text)
    if not bbox:
        return (0, 0)

    width = int(self.getlength(text))
    height = int(bbox[3])
    return (width, height)


if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    ImageFont.FreeTypeFont.getsize = custom_getsize  # type: ignore


class TxtRecognitionGenerator:
    def __init__(self) -> None:
        self._load_system_fonts()

        self.dataset_dir = DATA_DIR / "words-dataset"
        nltk.download("floresta", download_dir=DATA_DIR / "floresta")

    def _load_system_fonts(self):
        system_fonts_path = "C:\\Windows\\Fonts"
        self.available_fonts = glob.glob(
            os.path.join(system_fonts_path, "*.ttf")
        ) + glob.glob(os.path.join(system_fonts_path, "*.otf"))

    def apply_mask(self, mask: str):
        return "".join(
            str(random.randint(0, 9)) if char == "#" else char for char in mask
        )

    def generate_images_masked(self, count=1, blur=False, text_size=32, train=True):
        masks = [
            "R$ #,##",
            "R$ ##,##",
            "R$ ###,##",
            "##/##/2026",
            "(##) 9####-####",
            "(##)9####-####",
            "(##) ####-####",
            "###.###.###-##",
        ]

        content = []

        for i in range(count):
            for mask in masks:
                content.append(self.apply_mask(mask))

        random.shuffle(content)

        self.persist_images(content, blur, text_size, train)

    def generate_images(
        self, count=1, words_by_image=1, blur=False, text_size=32, train=True
    ):
        words = floresta.words()
        portuguese_words: list[str] = []

        word_count = count * words_by_image
        for w in words:
            if len(portuguese_words) >= word_count:
                break

            if isinstance(w, str) and w.isalpha() and len(w) >= 2:
                portuguese_words.append(w)

        random.shuffle(portuguese_words)

        content = [portuguese_words[0]]
        for w in portuguese_words[1:]:
            curr_words_count = len(content[-1].split(" "))

            if curr_words_count < words_by_image:
                content[-1] += f" {w}"
            else:
                content[-1] = self.random_case(content[-1])
                content.append(w)

        self.persist_images(content, blur, text_size, train)

    def persist_images(self, content, blur, text_size, train):
        generator = GeneratorFromStrings(
            strings=content,
            fonts=self.available_fonts,
            blur=1 if blur else 0,
            random_blur=blur,
            random_skew=True,
            count=len(content),
            size=text_size,
            fit=True,
            rtl=True,
        )

        output_dir = self.dataset_dir / ("train" if train else "test")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_dir / "labels.txt", "a", encoding="utf-8") as f:
            curr_number = self.get_last_number(output_dir)

            for i, (img, label) in enumerate(generator):
                index = curr_number + i
                img_name = f"{index}.png"
                img_path = os.path.join(output_dir / "images", img_name)
                img.save(img_path)
                f.write(f"images/{img_name}\t{label}\n")

        print(f"Generated {len(content)} base words in portuguese!")

    def random_case(self, word: str):
        r = random.random()
        if r < 0.33:
            return word.lower()
        elif r < 0.66:
            return word.capitalize()
        else:
            return word.upper()

    def get_last_number(self, dir):
        with open(dir / "labels.txt", "rb") as f:
            try:  # catch OSError in case of a one line file
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()

            if last_line != "":
                image_path = dir / last_line.split("\t")[0]
                name = image_path.name.split(".png")[0]

                return (int(name) + 1) if name.isdigit() else 0

        return 0


if __name__ == "__main__":
    trg = TxtRecognitionGenerator()
    for i in range(3):
        trg.generate_images(words_by_image=i + 1, count=100, train=False)

    trg.generate_images_masked(count=3, blur=True, train=False)
