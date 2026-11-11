import csv
import json
import random
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw as ID
from PIL.ImageFont import ImageFont as IF

from src.search_block.analyzer import Box
from src.utils.paths import DATA_DIR, test_image_path


class ContentLine:
    def __init__(self, text: str = ""):
        self.text = text
        self.bbox: Box

    def set_bbox(self, x, y, w, h):
        self.bbox = Box(x, y, w, h)

    def to_dict(self):
        return {"text": self.text, "bbox": self.bbox.to_dict()}


class PageGenerator:
    def __init__(self, fonts: list | None = None):
        # defaults to the catalog written by the CRNN generator
        self.fonts = fonts if fonts is not None else self.get_fonts()

    def generate_sentence(self) -> str:
        fake = Faker("pt_BR")

        text = fake.sentence(nb_words=random.randint(12, 40))

        return self.inject_accents(text)

    def inject_accents(self, text):
        substitutions = {
            "a": ["á", "à", "ã", "â"],
            "e": ["é", "ê"],
            "i": ["í"],
            "o": ["ó", "ô", "õ"],
            "u": ["ú"],
            "c": ["ç"],
        }
        text_list = list(text)
        # Change about 15% of eligible letters to include accents
        for i, char in enumerate(text_list):
            if char.lower() in substitutions and random.random() < 0.15:
                new_char = random.choice(substitutions[char.lower()])
                text_list[i] = new_char.upper() if char.isupper() else new_char
        return "".join(text_list)

    def get_fonts(self) -> list[Path]:
        fonts = []

        metadata_dir = DATA_DIR / "words-dataset" / "metadata"
        catalog_path = metadata_dir / "train_fonts.csv"
        with catalog_path.open("r", encoding="utf-8", newline="") as catalog_file:
            reader = csv.reader(catalog_file)
            for line in reader:
                _, font_path = line
                fonts.append(Path(font_path))

        return fonts

    def render(self, paragraphs: list[str], margins=(10, 10, 10, 10), font_path=None):
        width, height = self.mm_to_px(210), self.mm_to_px(297)  # 210 x 297mm
        if font_path is None:
            font_path = random.choice(self.fonts)

        x, y = self.mm_to_px(margins[0]), self.mm_to_px(margins[1])
        x2, y2 = width - self.mm_to_px(margins[2]), height - self.mm_to_px(margins[3])

        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)

        font_size = self.pt_to_px(random.randint(12, 25))
        font = ImageFont.truetype(font_path, size=font_size)

        # space = int(font_size * 0.8)
        spacing = int(font_size * 0.1)
        line_height = int(font_size * 1.2)

        curr_x, curr_y = x, y
        lines: list[ContentLine] = []
        for sentence in paragraphs:
            words = sentence.split(" ")

            line_text = []
            for idx in range(len(words)):
                line_text.append(words[idx])
                phrase = " ".join(line_text)
                left, top, right, bottom = draw.textbbox(
                    (curr_x, curr_y),
                    phrase,
                    font=font,
                    spacing=spacing,
                )

                greater = right >= x2
                is_last_word = idx == len(words) - 1

                if greater or is_last_word:
                    if greater and not is_last_word:
                        line_text = line_text[:-1]

                    phrase = " ".join(line_text)

                    if not is_last_word:
                        line_text = [words[idx]]

                    left, top, right, bottom = draw.textbbox(
                        (curr_x, curr_y),
                        phrase,
                        font=font,
                        spacing=spacing,
                    )

                    # ink box returned by textbbox, not the draw anchor,
                    # so crops never clip ascenders/descenders
                    line = ContentLine(phrase)
                    line.set_bbox(left, top, right - left, bottom - top)
                    lines.append(line)

                    draw.text(
                        (curr_x, curr_y), phrase, fill=0, font=font, spacing=spacing
                    )

                    curr_x = x
                    curr_y += line_height + spacing
                    line_text = []

                    if curr_y > y2:
                        break

            curr_y += line_height + spacing
            if curr_y > y2:
                break

        return image, lines, font_path

    def generate(
        self, paragraphs: list[str], margins=(10, 10, 10, 10), font_path=None
    ):
        image, lines, _ = self.render(paragraphs, margins, font_path)

        file_name = f"{random.randint(1000, 99999)}"
        image_path = test_image_path(f"{file_name}.png")
        image_path.parent.mkdir(parents=True, exist_ok=True, mode=777)

        json_file = test_image_path(f"{file_name}.json")
        clues = [line.to_dict() for line in lines]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"clues": clues}, f, indent=4)

        image.save(image_path)

        return image_path, clues

    @staticmethod
    def mm_to_px(mm: float, dpi: int = 150) -> int:
        pixels = (mm * dpi) / 25.4
        return round(pixels)

    @staticmethod
    def pt_to_px(pt: float, dpi: int = 150) -> int:
        pixels = (pt * dpi) / 72
        return round(pixels)


if __name__ == "__main__":
    pages_count = 3
    generator = PageGenerator()
    for _ in range(pages_count):
        paragraphs = []
        for _ in range(6):
            paragraphs.append(generator.generate_sentence())

        img_name, lines = generator.generate(paragraphs)
        print(img_name)
