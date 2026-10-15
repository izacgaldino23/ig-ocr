from torch import nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        
        # achata a imagem de 28x28 (bidimensional) para 784 (um vetor unidimensional)
        self.flatten = nn.Flatten()
        
        # Camada oculta
        self.fc1 = nn.Linear(28*28, 128)
        
        # função de ativação
        self.relu = nn.ReLU()
        
        # Camada de saida
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x