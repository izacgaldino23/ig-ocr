from torch import nn

MODELS = [[28 * 28, 128, 64, 10]]


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
        self.fc2 = nn.Linear(64, 10)

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
