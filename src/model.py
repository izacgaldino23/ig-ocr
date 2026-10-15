from torch import nn

MODELS = [
    [28*28, 128, 64, 10]
]

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, model_idx=0):
        super(SimpleNeuralNetwork, self).__init__()

        if not 0 <= model_idx < len(MODELS):
            raise ValueError(f"Índice de modelo inválido: {model_idx}")

        model_layers = MODELS[model_idx]
        
        self.flatten = nn.Flatten()
        
        # Lista para armazenar dinamicamente as camadas
        layers = []
        for i in range(len(model_layers) - 1):
            # Adiciona a camada Linear
            layers.append(nn.Linear(model_layers[i], model_layers[i+1]))
            
            # Adiciona a ReLU em todas as camadas, exceto na última (saída)
            if i < len(model_layers) - 2:
                layers.append(nn.ReLU())
                
        # Empacota tudo em um Sequential
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x