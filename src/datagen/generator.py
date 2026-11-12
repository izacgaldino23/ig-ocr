import csv
import glob
import hashlib
import io
import os
import random
import shutil

import nltk
from fontTools.ttLib import TTFont, TTLibError
from nltk.corpus import floresta
from PIL import ImageFont
from PIL.Image import Image
from trdg.generators import GeneratorFromStrings
from webdataset.writer import TarWriter

from src.datagen.page_generator import PageGenerator
from src.ocr.model import ALPHABET, CHAR2IDX
from src.utils.paths import DATA_DIR

FONT_STYLES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
REQUIRED_FONT_STYLES = frozenset(FONT_STYLES)
EXCLUDED_FONT_FAMILIES = {"garamond", "monotype corsiva"}


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

        font_records = []
        styles_by_family = {}
        family_names = {}
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
                    name_table = tt_font["name"]
                    family_name = name_table.getDebugName(
                        16
                    ) or name_table.getDebugName(1)
                if not family_name or not required_codepoints.issubset(cmap):
                    continue

                family_key = family_name.strip().casefold()
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
                font_records.append((font_path, family_key, style))
                styles_by_family.setdefault(family_key, set()).add(style)
                family_names[family_key] = family_name.strip()
            except (OSError, TTLibError, KeyError, ValueError) as exc:
                print(f"font {font_path} invalid: {exc}")

        allowed_families = {
            family
            for family, styles in styles_by_family.items()
            if REQUIRED_FONT_STYLES.issubset(styles)
            and family not in EXCLUDED_FONT_FAMILIES
        }
        if not allowed_families:
            raise RuntimeError("No complete font families support the OCR alphabet")

        self.available_fonts_by_style = {
            style: [
                font_path
                for font_path, family, font_style in font_records
                if family in allowed_families and font_style == style
            ]
            for style in FONT_STYLES
        }
        self.available_fonts = [
            font_path
            for fonts in self.available_fonts_by_style.values()
            for font_path in fonts
        ]
        self.available_font_families = sorted(
            family_names[family] for family in allowed_families
        )
        print(
            f"Using {len(self.available_font_families)} complete font families: "
            f"{', '.join(self.available_font_families)}"
        )

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

    def add_punctuation(self, line: str):
        # page lines are too few to teach quotes/dashes; the word set carries them
        roll = random.random()
        if roll < 0.10:
            return f"“{line}”"
        if roll < 0.14:
            return f"‘{line}’"
        if roll < 0.22:
            words = line.split(" ")
            if len(words) > 2:
                idx = random.randint(1, len(words) - 1)
                dash = random.choice([" - ", " – ", " — "])
                return " ".join(words[:idx]) + dash + " ".join(words[idx:])
        return line

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
            "R$ ##,##",
            "R$ ###,##",
            "US$ ##,##",
            "##/##/2026",
            "##-##-2026",
            "(##) 9####-####",
            "(##) ####-####",
            "###.###.###-##",
            "##.###.###/####-##",
            # ordinal and colon context: official docs are full of these
            "nº ####/####",
            "Art. #º",
            "#ª Vara Cível",
            "Tel: (##) 9####-####",
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

        content[-1] = self.random_case(content[-1])

        content = [self.add_punctuation(line) for line in content]

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

    def generate_page_lines(self, pages=1, train=True, clear=False, text="floresta"):
        page_generator = PageGenerator(fonts=self.available_fonts)
        samples = []

        for _ in range(pages):
            if text == "floresta":
                paragraphs = [page_generator.floresta_sentence() for _ in range(6)]
            else:
                paragraphs = [page_generator.generate_sentence() for _ in range(6)]

            # short field lines cover isolated values and the document chars;
            # they go first, otherwise the full page drops them
            if random.random() < 0.25:
                paragraphs = [
                    page_generator.document_line() for _ in range(random.randint(3, 5))
                ] + paragraphs

            image, lines, font_path = page_generator.render(paragraphs)

            for line in lines:
                # keep labels aligned with the model alphabet (OCRDataset
                # silently drops chars outside it, corrupting the target)
                if any(char not in CHAR2IDX for char in line.text):
                    continue
                x, y, w, h = line.bbox.get_values()
                samples.append((image.crop((x, y, x + w, y + h)), line.text, font_path))

        self._write_shard(samples, train, clear)

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
        fonts = self._fonts_for_style(bold, italic)
        generator = GeneratorFromStrings(
            strings=content,
            fonts=fonts,
            blur=1 if blur else 0,
            random_blur=blur,
            random_skew=True,
            count=len(content),
            size=text_size,
            fit=True,
            rtl=True,
            character_spacing=letter_spacing,
        )

        samples = (
            (img, label, fonts[generated_index % len(fonts)])
            for generated_index, (img, label) in enumerate(generator)
            if isinstance(img, Image)
        )

        self._write_shard(samples, train, clear)

    def _write_shard(self, samples, train, clear=False):
        output_dir = self.dataset_dir / ("train" if train else "test")
        split = "train" if train else "test"
        metadata_dir = self.dataset_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        usage_path = metadata_dir / f"{split}_font_usage.csv"
        catalog_path = metadata_dir / f"{split}_fonts.csv"

        if clear and output_dir.exists():
            shutil.rmtree(output_dir)
        if clear:
            usage_path.unlink(missing_ok=True)
            catalog_path.unlink(missing_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        shard_index = (
            max(
                (
                    int(path.stem.removeprefix("shard-"))
                    for path in output_dir.glob("shard-*.tar")
                ),
                default=-1,
            )
            + 1
        )
        shard_path = output_dir / f"shard-{shard_index:06d}.tar"
        temporary_path = output_dir / f".shard-{shard_index:06d}.tar.tmp"
        count = 0
        font_usage = []
        font_catalog = {}

        with TarWriter(f"file:{temporary_path.resolve().as_posix()}") as sink:
            for img, label, font_path in samples:
                font_id = hashlib.sha256(
                    os.path.normcase(os.path.abspath(font_path)).encode("utf-8")
                ).hexdigest()[:12]

                font_catalog[font_id] = font_path
                img_gray = img.convert("L")
                img_buffer = io.BytesIO()
                img_gray.save(img_buffer, format="PNG", optimize=True)

                sample_key = f"{shard_index:06d}_{count:09d}"
                sink.write(
                    {
                        "__key__": sample_key,
                        "png": img_buffer.getvalue(),
                        "txt": label,
                    }
                )
                font_usage.append((sample_key, font_id))
                count += 1

        if count:
            temporary_path.replace(shard_path)
            catalog_ids = set()
            if catalog_path.exists():
                with catalog_path.open(encoding="utf-8", newline="") as catalog_file:
                    catalog_ids = {
                        row["font_id"] for row in csv.DictReader(catalog_file)
                    }
            with catalog_path.open("a", encoding="utf-8", newline="") as catalog_file:
                writer = csv.writer(catalog_file)
                if catalog_file.tell() == 0:
                    writer.writerow(("font_id", "font_path"))
                for font_id, font_path in font_catalog.items():
                    if font_id not in catalog_ids:
                        writer.writerow((font_id, font_path))
            with usage_path.open("a", encoding="utf-8", newline="") as usage_file:
                writer = csv.writer(usage_file)
                if usage_file.tell() == 0:
                    writer.writerow(("sample_key", "font_id"))
                writer.writerows(font_usage)
        else:
            temporary_path.unlink()

        print(f"Added {count} samples into WebDataset shard {shard_path}")

    def random_case(self, word: str):
        # one style per sample, close to how documents are written:
        # mostly lowercase, some sentence case, few headings in caps
        r = random.random()
        if r < 0.60:
            return word.lower()
        elif r < 0.85:
            return word.capitalize()
        else:
            return word.upper()


if __name__ == "__main__":
    trg = TxtRecognitionGenerator()

    text_size = 32
    blur = False

    words_by_image = [i + 1 for i in range(8)]
    styles = [
        {"letter_spacing": 0, "bold": False, "italic": False},
        {"letter_spacing": 0, "bold": True, "italic": False},
        {"letter_spacing": 0, "bold": False, "italic": True},
        # letter_spacing=1 taught the model to insert spaces inside words
        {"letter_spacing": 0, "bold": True, "italic": True},
    ]

    words_count = 300
    custom_count = 75
    page_count = 90

    for train in [False, True]:
        words = words_count if train else int(words_count / 3)
        custom = custom_count if train else int(custom_count / 3)  # x * 8
        pages = page_count if train else int(page_count / 3)
        clear = True
        for style_idx, style in enumerate(styles):
            for i, qtd in enumerate(words_by_image):
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

        trg.generate_page_lines(pages=pages, train=train)
