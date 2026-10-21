import json
import random

from PIL import Image, ImageDraw, ImageFont

from src.char_reader_nn.paths import test_image_path


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
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    clue = []

    curr_x, curr_y = (
        random.randint(margin, int(width / 2)),
        random.randint(margin, int(height / 2)),
    )
    initial_x = curr_x

    words = text.split(r" ")

    for word in words:
        left, top, right, bottom = draw.textbbox((curr_x, curr_y), word, font=font)
        if right >= right_limit:
            curr_x = initial_x
            curr_y += 60

        for char in word:
            if char == "\n":  # new line
                curr_x = initial_x
                curr_y += 60
                continue

            if (
                curr_x + 20 >= right_limit
            ):  # brake line to prevent writing letters out of the image
                curr_x = initial_x
                curr_y += 60

            # get the bounding box for each characther
            bbox = draw.textbbox((curr_x, curr_y), char, font=font)
            left, top, right, bottom = bbox
            w, h = right - left, bottom - top

            # Draw a white characther
            draw.text((curr_x, curr_y), char, fill=255, font=font)

            # save the char bounding box on clue
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
    image_file.parent.mkdir(parents=True, exist_ok=True)

    image.save(image_file)
    with open(json_file, "w") as f:
        json.dump({"expected": text, "box": clue}, f, indent=4)

    print(f"✅ Image: {image_file} | Clue: {json_file}")


# # Exemplo de teste com duas linhas e espaço
# gerar_imagem_teste("A1 B2\ni 7")
if __name__ == "__main__":
    generate_test_image("Esse é um teste bem massa")
