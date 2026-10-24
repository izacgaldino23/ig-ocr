import cv2
import numpy as np
import torch

from src.neural_network.config import init_config
from src.neural_network.model import MEAN, STD, SimpleCNN
from src.search_block.analyzer import character_segmentation
from src.utils.paths import checkpoint_path


# process_roi receives a cut of the original image of each character
# stabilize it and return a 28x28 tensor
# ready for EMINIST Neural Network processing
# ROI means Region Of Interest
def process_roi(roi: np.ndarray) -> torch.Tensor:
    h, w = roi.shape

    # get the bigger side
    max_side = max(h, w)

    # add a padding rounding the characther
    padding = int(max_side * 0.2)
    square_size = max_side + 2 * padding

    # create a black square image
    square_img = np.zeros((square_size, square_size), dtype=np.uint8)

    # get central point to paste the cutted character
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2
    square_img[y_offset : y_offset + h, x_offset : x_offset + w] = roi

    # resize to 28x28, EMINIST default size
    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

    # convert to tensor pytorch and normalize (0 to 1)
    tensor_img = torch.from_numpy(resized).float() / 255.0

    # add chanel and batch dimensions: [1, 1, 28, 28]
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)

    # identical normalization with the training
    tensor_img = (tensor_img - MEAN) / STD

    return tensor_img


if __name__ == "__main__":
    img, lines = character_segmentation("6455")
    original_text = (
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry"
    )
    predicted_text = ""

    args, device = init_config(True)

    model = SimpleCNN()
    file = checkpoint_path(args.input)
    content = torch.load(file)
    model.load_state_dict(content["model"])
    class_names: list[str] = content["classes"]

    model = model.to(device)

    # desligando o calculo de gradientes
    model.eval()
    tollerance = 4
    new_line = False

    for line in lines:
        curr_x = -1
        for x, y, w, h in line:
            roi = img[y : y + h, x : x + w]
            input_tensor = process_roi(roi)

            if new_line:
                new_line = False
                predicted_text += " "

            if curr_x != -1 and abs(x - curr_x) > tollerance:  # has space
                print(f"Space size {abs(x - curr_x)}")
                predicted_text += " "

            curr_x = x + w
            curr_y = y

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                char = class_names[int(pred.item())]

                predicted_text += char

        new_line = True

    print(f"Original text:  \t '{original_text}'")
    print(f"Predicted text: \t '{predicted_text}'")
