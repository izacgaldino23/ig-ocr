import glob
import io
import os
import random
import shutil

import lmdb
import nltk
from fontTools.ttLib import TTFont, TTLibError
from nltk.corpus import floresta
from PIL import ImageFont
from PIL.Image import Image
from trdg.generators import GeneratorFromStrings

from src.crnn.model import ALPHABET
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
        required_codepoints = {ord(char) for char in ALPHABET}

        valid_fonts = []
        valid_fonts_by_style = {
            (bold, italic): [] for bold in (False, True) for italic in (False, True)
        }
        for font_path in all_fonts:
            basename = os.path.basename(font_path).lower()
            if any(exc in basename for exc in excluded_keywords):
                continue
            try:
                font = ImageFont.truetype(font_path, 32)
                if font.getlength("a") <= 0:
                    continue
                with TTFont(font_path, lazy=True) as tt_font:
                    cmap = tt_font.getBestCmap() or {}
                    os2 = tt_font.get("OS/2")
                    head = tt_font.get("head")
                if required_codepoints.issubset(cmap):
                    valid_fonts.append(font_path)
                    style = (
                        bool(
                            (os2 and os2.fsSelection & (1 << 5))  # type: ignore
                            or (head and head.macStyle & 1)  # type: ignore
                        ),
                        bool(
                            (os2 and os2.fsSelection & 1)  # type: ignore
                            or (head and head.macStyle & 2)  # type: ignore
                        ),
                    )
                    valid_fonts_by_style[style].append(font_path)
            except (OSError, TTLibError, KeyError, ValueError) as exc:
                print(f"font {font_path} invalid: {exc}")

        if not valid_fonts:
            raise RuntimeError("No system fonts support the complete OCR alphabet")

        self.available_fonts = valid_fonts
        self.available_fonts_by_style = valid_fonts_by_style

    def _fonts_for_style(self, bold=None, italic=None):
        fonts = [
            font
            for (
                font_bold,
                font_italic,
            ), style_fonts in self.available_fonts_by_style.items()
            if (bold is None or bold == font_bold)
            and (italic is None or italic == font_italic)
            for font in style_fonts
        ]
        if not fonts:
            raise RuntimeError(
                f"No compatible fonts found for bold={bold}, italic={italic}"
            )
        return fonts

    def apply_mask(self, mask: str):
        return "".join(
            str(random.randint(0, 9)) if char == "#" else char for char in mask
        )

    def generate_images_masked(
        self,
        count=1,
        blur=False,
        text_size=32,
        train=True,
        clear=False,
        letter_spacing=0,
        bold=None,
        italic=None,
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

        self.persist_images(
            content,
            blur,
            text_size,
            train,
            clear=clear,
            letter_spacing=letter_spacing,
            bold=bold,
            italic=italic,
        )

    def generate_images(
        self,
        count=1,
        words_by_image=1,
        blur=False,
        text_size=32,
        train=True,
        clear=False,
        letter_spacing=0,
        bold=None,
        italic=None,
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

        self.persist_images(
            content,
            blur,
            text_size,
            train,
            clear=clear,
            letter_spacing=letter_spacing,
            bold=bold,
            italic=italic,
        )

    def persist_images(
        self,
        content,
        blur,
        text_size,
        train,
        clear=False,
        letter_spacing=0,
        bold=None,
        italic=None,
    ):
        output_dir = self.dataset_dir / ("train" if train else "test")
        lmdb_path = str(output_dir / "lmdb_data")

        if clear and output_dir.exists():
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        generator = GeneratorFromStrings(
            strings=content,
            fonts=self._fonts_for_style(bold, italic),
            blur=1 if blur else 0,
            random_blur=blur,
            random_skew=True,
            count=len(content),
            size=text_size,
            fit=True,
            rtl=True,
            character_spacing=letter_spacing,
        )

        map_size = 500 * 1024 * 1024
        env = lmdb.open(lmdb_path, map_size=map_size)

        with env.begin(write=False) as txn:
            num_samples_ptr = txn.get(b"num-samples")
            curr_number = (
                0
                if (clear or num_samples_ptr is None)
                else int(num_samples_ptr.decode("utf-8"))
            )

        txn = env.begin(write=True)
        count = 0

        for i, (img, label) in enumerate(generator):
            if not isinstance(img, Image):
                continue

            index = curr_number + count + 1

            img_gray = img.convert("L")
            img_buffer = io.BytesIO()
            img_gray.save(img_buffer, format="PNG", optimize=True)
            img_bytes = img_buffer.getvalue()

            image_key = f"image-{index:09d}".encode()
            label_key = f"label-{index:09d}".encode()

            txn.put(image_key, img_bytes)
            txn.put(label_key, label.encode("utf-8"))

            count += 1

            if count % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

        total_samples = curr_number + count
        txn.put(b"num-samples", str(total_samples).encode("utf-8"))
        txn.commit()
        env.close()

        print(f"Added {count} samples into LMDB on {lmdb_path}. Total: {total_samples}")

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

    text_size = 32
    blur = False

    words_by_image = [i + 1 for i in range(8)]
    styles = [
        {"letter_spacing": 0, "bold": False, "italic": False},
        {"letter_spacing": 0, "bold": True, "italic": False},
        {"letter_spacing": 0, "bold": False, "italic": True},
        {"letter_spacing": 1, "bold": True, "italic": True},
    ]

    words_count = 300
    custom_count = 75

    for train in [True, False]:
        words = words_count if train else int(words_count / 3)
        custom = custom_count if train else int(custom_count / 3)  # x * 8
        clear = True
        for style_idx, style in enumerate(styles):
            for i, qtd in enumerate(words_by_image):
                # word_count = words // len(styles) + (style_idx < words % len(styles))
                # custom_count = custom // len(styles) + (
                #     style_idx < custom % len(styles)
                # )
                trg.generate_images(
                    words_by_image=qtd,
                    count=words,
                    blur=blur,
                    train=train,
                    text_size=text_size,
                    clear=clear,
                    **style,
                )
                clear = False
            trg.generate_images_masked(
                count=custom,
                blur=blur,
                train=train,
                text_size=text_size,
                **style,
            )
