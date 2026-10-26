import matplotlib.pyplot as plt
import torch

from src.neural_network.config import init_config
from src.neural_network.model import (
    MEAN,
    STD,
    get_emnist_dataset,
    model_from_checkpoint,
)
from src.utils.paths import checkpoint_path

# 1. Initialize configuration.
args, device = init_config(True)

# 2. Download and load the dataset.
test_dataset = get_emnist_dataset(False)

# 3. Model.
# model = SimpleNeuralNetwork(args.input)
file = checkpoint_path(args.input)
content = torch.load(file)
class_names = content["classes"]
model = model_from_checkpoint(content).to(device)

# Disable gradient calculation.
model.eval()

# Select a few images to visualize.
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

with torch.no_grad():
    # the first 5 wrong predictions
    wrong = 0
    i = 0
    for image, label in test_dataset:
        # Add a batch dimension; the network expects [batch_size, channels, height, width].
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        predict_idx = int(predicted.item())
        predicted_item = class_names[predict_idx]

        real_label = class_names[label]
        if predict_idx != label:
            # Prepare the image for display by undoing normalization.
            img_display = image.squeeze(0) * STD + MEAN

            axes[wrong].imshow(img_display, cmap="gray")
            axes[wrong].set_title(f"Actual: {real_label}\nPredicted: {predicted_item}")
            print(f"Actual: {real_label}\t Predicted: {predicted_item}\t Index: {i}")
            axes[wrong].axis("off")

            wrong += 1

        if wrong == 5 or i == len(test_dataset) - 1:
            break
        i += 1

plt.tight_layout()
plt.show()
