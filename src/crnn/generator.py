import glob
import os
import random
import shutil

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
        all_fonts = glob.glob(os.path.join(system_fonts_path, "*.ttf")) + glob.glob(
            os.path.join(system_fonts_path, "*.otf")
        )

        # Filter out fonts known to be symbols, emojis, or lacking standard characters
        excluded_keywords = [
            "symbol",
            "bssym",
            "wingdings",
            "webdings",
            "marlett",
            "emoji",
            "codec",
            "math",
            "david",
            "mangal",
            "leelawadee",
        ]

        valid_fonts = []
        for font_path in all_fonts:
            basename = os.path.basename(font_path).lower()
            if any(exc in basename for exc in excluded_keywords):
                continue
            try:
                # Test loading the font and getting length for basic Latin/Portuguese chars
                font = ImageFont.truetype(font_path, 32)
                if font.getlength("a") > 0:
                    valid_fonts.append(font_path)
            except Exception:
                print(f"font {font_path} invalid")
                continue

        self.available_fonts = valid_fonts if valid_fonts else all_fonts

    def apply_mask(self, mask: str):
        return "".join(
            str(random.randint(0, 9)) if char == "#" else char for char in mask
        )

    def generate_images_masked(
        self, count=1, blur=False, text_size=32, train=True, clear=False
    ):
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

        # random.shuffle(content)

        self.persist_images(content, blur, text_size, train, clear=clear)

    def generate_images(
        self,
        count=1,
        words_by_image=1,
        blur=False,
        text_size=32,
        train=True,
        clear=False,
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

        self.persist_images(content, blur, text_size, train, clear=clear)

    def persist_images(self, content, blur, text_size, train, clear=False):
        output_dir = self.dataset_dir / ("train" if train else "test")
        images_dir = output_dir / "images"

        if clear:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.makedirs(images_dir, exist_ok=True)

        os.makedirs(images_dir, exist_ok=True)

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

        labels_file = output_dir / "labels.txt"
        curr_number = (
            0 if clear or not labels_file.exists() else self.get_last_number(output_dir)
        )

        mode = "w" if clear or not labels_file.exists() else "a"
        with open(labels_file, mode, encoding="utf-8") as f:
            for i, (img, label) in enumerate(generator):
                index = curr_number + i
                img_name = f"{index}.png"
                img_path = os.path.join(images_dir, img_name)

                # Convert image to grayscale ("L") and optimize to keep file size very small
                img_gray = img.convert("L")
                img_gray.save(img_path, optimize=True)
                f.write(f"images/{img_name}\t{label}\n")

        print(f"Generated {len(content)} images in {output_dir}!")

    def random_case(self, word: str):
        r = random.random()
        if r < 0.33:
            return word.lower()
        elif r < 0.66:
            return word.capitalize()
        else:
            return word.upper()

    def get_last_number(self, dir):
        labels_path = dir / "labels.txt"
        if not labels_path.exists():
            return 0
        with open(labels_path, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()

            if last_line != "":
                parts = last_line.split("\t")
                if len(parts) > 0:
                    image_path = dir / parts[0]
                    name = image_path.name.split(".png")[0]

                    return (int(name) + 1) if name.isdigit() else 0

        return 0


if __name__ == "__main__":
    trg = TxtRecognitionGenerator()
    for i in range(5):
        trg.generate_images(
            words_by_image=i + 1, count=100, train=False, clear=(i == 0), text_size=20
        )

    trg.generate_images_masked(
        count=5, blur=True, train=False, clear=False, text_size=20
    )
