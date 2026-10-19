import torch
from torch.utils.data import DataLoader

from .args import get_args
from .model import SimpleCNN, get_emnist_dataset
from .paths import checkpoint_path

args = get_args()

# Baixar e carregar o conjunto de dados
test_dataset = get_emnist_dataset(False, False)
print("dataset carregado com sucesso!")

# Criar o dataloader para separar os dados em lotes
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# modelo
# model = SimpleNeuralNetwork(args.input)
model = SimpleCNN()
file = checkpoint_path(args.input)
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
