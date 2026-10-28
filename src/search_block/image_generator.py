import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.utils.paths import test_image_path


def _find_shantell_font() -> Path | None:
    font_dirs = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    windows_dir = os.environ.get("WINDIR")

    if local_app_data:
        font_dirs.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
    if windows_dir:
        font_dirs.append(Path(windows_dir) / "Fonts")

    for font_dir in font_dirs:
        if font_dir.is_dir():
            for font_file in font_dir.iterdir():
                if (
                    font_file.suffix.casefold() in {".ttf", ".otf"}
                    and "shantell" in font_file.stem.casefold()
                ):
                    return font_file

    return None


def generate_test_image(text: str):
    # random generate image size
    minSize, maxSize = random.randint(200, 300), random.randint(500, 600)
    width, height = random.randint(minSize, maxSize), random.randint(minSize, maxSize)

    margin = 20
    right_limit = width - margin

    # generate image
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)

    # try to load a system font
    font_size = random.randint(10, 25)
    font_path = _find_shantell_font()
    if font_path is None:
        font = ImageFont.load_default()
        print("Shantell Sans font file not found; using the default font.")
    else:
        try:
            font = ImageFont.truetype(str(font_path), size=font_size)
        except OSError as error:
            font = ImageFont.load_default()
            print(f"Could not load Shantell Sans from {font_path}: {error}")

    clue = []

    curr_x, curr_y = (
        random.randint(margin, int(width / 2)),
        random.randint(margin, int(height / 2)),
    )
    initial_x = curr_x

    words = text.split(r" ")

    spacing = 0
    for word in words:
        left, top, right, bottom = draw.textbbox(
            (curr_x, curr_y), word, font=font, spacing=spacing
        )
        if right >= right_limit:
            curr_x = initial_x
            curr_y += 60
            # re math the new coordinates:
            left, top, right, bottom = draw.textbbox(
                (curr_x, curr_y), word, font=font, spacing=spacing
            )

        for char in word:
            if char == "\n" or char == "":  # new line
                if char == "\n":
                    curr_x = initial_x
                    curr_y += 60
                continue

            # get the bounding box for each characther
            bbox = draw.textbbox((curr_x, curr_y), char, font=font, spacing=spacing)
            left, top, right, bottom = bbox
            w, h = right - left, bottom - top

            # Draw a white characther
            draw.text((curr_x, curr_y), char, fill=255, font=font, spacing=spacing)

            # save the char bounding box on clue
            if char != " ":
                clue.append(
                    {
                        "char": char,
                        "bbox": [left, top, w, h],  # [x, y, largura, altura]
                    }
                )

            # Advance for the new letter possition
            curr_x += w + 2

        if curr_x != initial_x:  # space between words
            curr_x += font_size - 2
        continue

    file_name = f"{random.randint(1000, 9999)}"
    image_file = test_image_path(f"{file_name}.png")
    json_file = test_image_path(f"{file_name}.json")
    image_file.parent.mkdir(parents=True, exist_ok=True, mode=777)

    image.save(image_file)
    with open(json_file, "w") as f:
        json.dump({"expected": text, "box": clue}, f, indent=4)

    # print(f"✅ Image: {image_file} | Clue: {json_file}")
    return file_name


if __name__ == "__main__":
    # Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London, took a 1914 Cicero translation and scrambled it to make dummy text for Letraset's Body Type sheets. It has survived not only many decades, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised thanks to these sheets and more recently with desktop publishing software like Aldus PageMaker and Microsoft Word including versions of Lorem Ipsum.
    generate_test_image(
        # "Lorem Ipsum is simply dummy text of the printing and typesetting industry"
        # "Fifi found five fluffy foxes beside the little mill"
        "I like Lily and Mimi who use 10 old maps while Oliver and Sam read"
    )
