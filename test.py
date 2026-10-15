import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleNeuralNetwork
from args import get_args

args = get_args()

# 1. Transformação para converter as imagens em tensores e normaliza-las
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. Baixar e carregar o conjunto de dados
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 3. Criar o dataloader para separar os dados em lotes
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 4. modelo
model = SimpleNeuralNetwork(args.input)
file = f'data/models/mnist-{args.input}.pth'
model.load_state_dict(torch.load(file))

# desligando o calculo de gradientes
model.eval()
correct = 0
total = 0
# n: [total, correct]
number_map = {i: [0, 0] for i in range(10)}

with torch.no_grad():
    for image, label in test_loader:
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        for i in range(label.size(0)):
            number_map[label[i].item()][0] += 1
            if predicted[i].item() == label[i].item():
                number_map[label[i].item()][1] += 1
        
acurracy = correct / total
print(f"Acurácia da rede de imagens de testes: {acurracy:.4f}")

for i in range(10):
    print(f"{i}: {number_map[i][1] / number_map[i][0]:.4f}")