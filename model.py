from torch import nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        
        # achata a imagem de 28x28 (bidimensional) para 784 (um vetor unidimensional)
        self.flatten = nn.Flatten()
        
        # Camada oculta 1
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        
        # Camada oculta 2
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        
        # camada oculta 3
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        
        # Camada de saida
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        
        return x