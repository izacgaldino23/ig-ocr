import matplotlib.pyplot as plt
import torch

from .args import get_args
from .model import MEAN, STD, SimpleCNN, get_emnist_dataset
from .paths import checkpoint_path

args = get_args()

# 2. Baixar e carregar o conjunto de dados
test_dataset = get_emnist_dataset(False, False)

# 3. modelo
# model = SimpleNeuralNetwork(args.input)
model = SimpleCNN()
file = checkpoint_path(args.input)
model.load_state_dict(torch.load(file))

# desligando o calculo de gradientes
model.eval()

# escolher algumas imagens para visualizar
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

with torch.no_grad():
    # the first 5 wrong predictions
    wrong = 0
    i = 0
    for image, label in test_dataset:
        # Adiciona uma dimensão extra para o lote (batch) já que a rede espera [batch_size, canais, altura, largura]
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        if predicted.item() != label:
            # Prepara a imagem para exibição (removendo a normalização e convertendo para numpy)
            img_display = image.squeeze(0) * STD + MEAN

            axes[wrong].imshow(img_display, cmap="gray")
            axes[wrong].set_title(f"Real: {label}\nPred: {predicted.item()}")
            print(f"Real: {label}\t Pred: {predicted.item()}\t Index: {i}")
            axes[wrong].axis("off")

            wrong += 1

        if wrong == 5 or i == len(test_dataset) - 1:
            break
        i += 1

plt.tight_layout()
plt.show()
