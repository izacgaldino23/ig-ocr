import torch
from PIL import Image
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

from src.paths import DATA_DIR

MODELS = [[28 * 28, 128, 64, 10]]
# Valores de normalização do MNIST
MEAN = 0.1307
STD = 0.3081

EMINIST_BALANCED_VALUES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, model_idx=0):
        super().__init__()

        if not 0 <= model_idx < len(MODELS):
            raise ValueError(f"Índice de modelo inválido: {model_idx}")

        model_layers = MODELS[model_idx]

        self.flatten = nn.Flatten()

        # Lista para armazenar dinamicamente as camadas
        layers = []
        for i in range(len(model_layers) - 1):
            # Adiciona a camada Linear
            layers.append(nn.Linear(model_layers[i], model_layers[i + 1]))

            # Adiciona a ReLU em todas as camadas, exceto na última (saída)
            if i < len(model_layers) - 2:
                layers.append(nn.ReLU())

        # Empacota tudo em um Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Primeira camada convolucional + pooling
        # Entra [1, 28, 28] e sai [16, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # reduz pela metade [16, 28, 28] -> [16, 14, 14]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Segunda camada convolucional + pooling
        # Entra [16, 14, 14] e sai [32, 14, 14]
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        # reduz pela metade [32, 14, 14] -> [32, 7, 7]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3. Camadas completamente conectadas (Dense / Linear)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 47)

    def forward(self, x):
        # Bloco Convolucional 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Bloco Convolucional 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Transforma o volume [32, 7, 7] em um vetor de 1568 valores
        x = self.flatten(x)

        # Classificação Final
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


def custom_noise_erasing(img):
    if torch.rand(1).item() > 0.1:  # Probabilidade p=0.1
        return img

    # Calcula parâmetros da região cortada
    i, j, h, w, _ = v2.RandomErasing.get_params(
        img, scale=(0.01, 0.2), ratio=(0.2, 2.2), value=[0]
    )

    # Valores normalizados para preto (0.0) e cinza médio (0.5)
    val_black = (0.0 - MEAN) / STD  # ~ -0.424
    val_gray = (0.5 - MEAN) / STD  # ~  1.200

    # Gera ruído uniforme entre preto e cinza médio
    noise = torch.empty(
        (img.shape[0], h, w), dtype=img.dtype, device=img.device
    ).uniform_(val_black, val_gray)

    # Aplica o apagamento usando a API v2
    return v2.functional.erase(img, i, j, h, w, noise)


def fix_emnist_orientation(img):
    return img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)


def get_transformers():
    degrees = 10.0

    # Transformação para converter as imagens em tensores e normaliza-las
    # Adicionado também um random de posição, rotação e escala
    # Além de um custom noise erasing
    return v2.Compose(
        [
            v2.Lambda(fix_emnist_orientation),
            v2.RandomAffine(
                (-degrees, degrees), translate=(0.1, 0.1), scale=(0.8, 1.1)
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((MEAN,), (STD,)),
            v2.Lambda(custom_noise_erasing),
        ]
    )


def get_emnist_dataset(train: bool, download: bool):
    return datasets.EMNIST(
        DATA_DIR,
        "balanced",
        train=train,
        download=download,
        transform=get_transformers(),
    )
