import csv
import random
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from src.search_block.analyzer import Box
from src.utils.paths import DATA_DIR, test_image_path


class ContentLine:
    def __init__(self, text: str=""):
        self.text = text
        self.bbox: Box

    def set_bbox(self, x, y, w, h):
        self.bbox = Box(x, y, w, h)


class PageGenerator:
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

    def generate(self, paragraphs: list[str], margins=(10, 10, 10, 10)):
        width, height = self.mm_to_px(210), self.mm_to_px(297)  # 210 x 297mm
        fonts = self.get_fonts()

        x, y = self.mm_to_px(margins[0]), self.mm_to_px(margins[1])
        x2, y2 = width - self.mm_to_px(margins[2]), height - self.mm_to_px(margins[3])

        image = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(image)

        font_size = self.pt_to_px(random.randint(12, 25))
        random.shuffle(fonts)
        font_path = fonts[0]
        font = ImageFont.truetype(font_path, size=font_size)

        space = int(font_size * 0.8)
        spacing = int(font_size * 0.1)
        line_height = int(font_size * 1.2)

        curr_x, curr_y = x, y
        for sentence in paragraphs:
            words = sentence.split(" ")

            for word in words:
                left, top, right, bottom = draw.textbbox(
                    (curr_x, curr_y), word, font=font, spacing=spacing
                )

                if right > x2:
                    curr_x = x
                    curr_y += line_height + spacing

                    left, top, right, bottom = draw.textbbox(
                        (curr_x, curr_y), word, font=font, spacing=spacing
                    )
                    
                word_w = right - left

                if bottom > y2:
                    break

                draw.text((curr_x, curr_y), word, fill=0, font=font, spacing=spacing)
                curr_x += word_w + space

            curr_x, curr_y = x, curr_y + font_size * 2 + spacing
            if curr_y > y2:
                break

        file_name = f"{random.randint(1000, 9999)}"
        image_file = test_image_path(f"{file_name}.png")
        image_file.parent.mkdir(parents=True, exist_ok=True, mode=777)

        image.save(image_file)

        return file_name

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

        generator.generate(paragraphs)
