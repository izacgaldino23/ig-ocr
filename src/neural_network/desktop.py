import tkinter as tk

import torch
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms

from src.neural_network.config import init_config
from src.neural_network.model import MEAN, STD, model_from_checkpoint
from src.utils.paths import checkpoint_path

args, device = init_config()

BAR_WIDTH = 160
BAR_HEIGHT = 14
PREVIEW_SCALE = 4  # Enlarge the 28x28 image 4x for visual inspection.


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")

        # 1. Load the trained model.
        # self.model = SimpleNeuralNetwork(args.input)
        file = checkpoint_path(args.input)
        content = torch.load(file)
        self.class_names = content["classes"]
        self.model = model_from_checkpoint(content).to(device)

        self.model.eval()

        # 2. Left column: fixed drawing area.
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_size = 280
        self.canvas = tk.Canvas(
            left, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Bind the mouse-drag event for drawing.
        buttons = tk.Frame(left)
        buttons.pack(pady=10)

        self.btn_predict = tk.Button(
            buttons, text="Predict", command=self.predict
        )
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(buttons, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 3. Pillow image used to store the drawing at high resolution.
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_tool = ImageDraw.Draw(self.image)

        # 4. Right column: 28x28 preview and probability list.
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.preview_label = tk.Label(right, text="28x28 Preview", font=("Arial", 11))
        self.preview_label.pack(anchor="w")

        self.label_result = tk.Label(
            right, text="Draw a character above", font=("Arial", 13)
        )
        self.label_result.pack(anchor="w", pady=(8, 4))

        rows = tk.Frame(right)
        rows.pack(anchor="w")
        self.digit_labels = []
        self.bar_canvases = []
        self.bar_rects = []
        for d in range(10):
            row = tk.Frame(rows)
            row.pack(anchor="w")
            lbl = tk.Label(row, text=str(d), width=2, font=("Arial", 12, "bold"))
            lbl.pack(side=tk.LEFT)
            canvas = tk.Canvas(
                row,
                width=BAR_WIDTH,
                height=BAR_HEIGHT,
                bg="#3a3a3a",
                highlightthickness=0,
            )
            canvas.pack(side=tk.LEFT, padx=6)
            rect = canvas.create_rectangle(0, 0, 0, BAR_HEIGHT, fill="#7f8c8d", width=0)
            self.digit_labels.append(lbl)
            self.bar_canvases.append(canvas)
            self.bar_rects.append(rect)

    def draw(self, event):
        # Draw a white circle along the mouse path.
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill="white", outline="white"
        )
        self.draw_tool.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_tool = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a character above")
        self.preview_label.config(image="", text="28x28 Preview")
        self.photo = None
        self._reset_bars()

    def _reset_bars(self):
        for i in range(10):
            self.digit_labels[i].config(fg="black")
            self.bar_canvases[i].coords(self.bar_rects[i], 0, 0, 0, BAR_HEIGHT)
            self.bar_canvases[i].itemconfig(self.bar_rects[i], fill="#7f8c8d")

    def predict(self):
        # Crop and center the character before resizing it to the MNIST format.
        img_processed = self.preprocess_and_center(self.image)

        # Enlarge the preview with nearest-neighbor interpolation to inspect its pixels.
        preview = img_processed.resize(
            (28 * PREVIEW_SCALE, 28 * PREVIEW_SCALE), Image.Resampling.NEAREST
        )
        self.photo = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=self.photo, text="")

        # Convert to a tensor and normalize.
        tensor_image = transforms.ToTensor()(img_processed)
        tensor_image = transforms.Normalize((MEAN,), (STD,))(tensor_image)
        tensor_image = tensor_image.unsqueeze(0)

        # Run inference and convert the logits to probabilities.
        with torch.no_grad():
            output = self.model(tensor_image)
            probs = torch.softmax(output, dim=1)[0]
            predicted = probs.argmax().item()

            predict_idx = int(predicted)
            predicted_item = self.class_names[predict_idx]

        best = predicted_item
        for d in range(10):
            p = probs[d].item()
            self.bar_canvases[d].coords(
                self.bar_rects[d], 0, 0, max(1, round(p * BAR_WIDTH)), BAR_HEIGHT
            )
            if d == best:
                self.bar_canvases[d].itemconfig(self.bar_rects[d], fill="#2ecc71")
                self.digit_labels[d].config(fg="#2ecc71")
            else:
                self.bar_canvases[d].itemconfig(self.bar_rects[d], fill="#7f8c8d")
                self.digit_labels[d].config(fg="black")

        self.label_result.config(text=f"Network prediction: {predicted_item}")

    def preprocess_and_center(self, image_pil):
        # 1. Find the bounding box of the drawn pixels.
        bbox = image_pil.getbbox()

        # Return the blank image if the canvas is empty.
        if not bbox:
            return image_pil.resize((28, 28))

        # Crop the image to contain only the character.
        digit_crop = image_pil.crop(bbox)

        # 2. Preserve the aspect ratio by placing the crop in a square.
        w, h = digit_crop.size
        max_side = max(w, h)

        # Create a new black square canvas with a small margin.
        square_img = Image.new("L", (max_side, max_side), color=0)

        # Paste the crop at the center of the square.
        offset = ((max_side - w) // 2, (max_side - h) // 2)
        square_img.paste(digit_crop, offset)

        # 3. Resize the character with MNIST-style margins (20x20 within 28x28).
        digit_20x20 = square_img.resize((20, 20), Image.Resampling.LANCZOS)

        # 4. Create the final 28x28 image and center the 20x20 character (4-pixel margin).
        final_28x28 = Image.new("L", (28, 28), color=0)
        final_28x28.paste(digit_20x20, (4, 4))

        return final_28x28


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
