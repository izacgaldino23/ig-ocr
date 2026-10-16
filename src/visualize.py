from numbers import Number

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import v2

from .args import get_args
from .model import SimpleCNN
from .paths import DATA_DIR, checkpoint_path

args = get_args()

# Valores de normalização do MNIST
mean = 0.1307
std = 0.3081

# Valores normalizados para preto (0.0) e cinza médio (0.5)
val_black = (0.0 - mean) / std  # ~ -0.424
val_gray = (0.5 - mean) / std   # ~  1.200
degrees = 10.0

def custom_noise_erasing(img):
    if torch.rand(1).item() > 0.1:  # Probabilidade p=0.1
        return img

    # Calcula parâmetros da região cortada
    i, j, h, w, _ = v2.RandomErasing.get_params(
        img, scale=(0.01, 0.2), ratio=(0.2, 2.2), value=[0]
    )

    # Gera ruído uniforme entre preto e cinza médio
    noise = torch.empty((img.shape[0], h, w), dtype=img.dtype, device=img.device).uniform_(val_black, val_gray)

    # Aplica o apagamento usando a API v2
    return v2.functional.erase(img, i, j, h, w, noise)

# 1. Transformação para converter as imagens em tensores e normaliza-las
transform = v2.Compose(
    [
        v2.RandomAffine((-degrees, degrees), translate=(0.1, 0.1), scale=(0.8, 1.1)),
        v2.ToTensor(),
        v2.Normalize((mean,), (std,)),
        v2.Lambda(custom_noise_erasing),
    ]
)

# 2. Baixar e carregar o conjunto de dados
test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform)

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
            img_display = image.squeeze(0) * std + mean

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
