import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleNeuralNetwork

# 1. Transformação para converter as imagens em tensores e normaliza-las
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. Baixar e carregar o conjunto de dados
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 3. Criar o dataloader para separar os dados em lotes
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 4. modelo
model = SimpleNeuralNetwork()
model.load_state_dict(torch.load('mnist.pth'))

# desligando o calculo de gradientes
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for image, labels in test_loader:
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
acurracy = correct / total
print(f"Acurácia da rede de imagens de testes: {acurracy:.4f}")