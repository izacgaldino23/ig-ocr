import torch
from torch.utils.data import DataLoader

from src.config import init_config

from .model import SimpleCNN, get_emnist_dataset
from .paths import checkpoint_path

# 1. Configuração inicial
args, device = init_config(True)

# Baixar e carregar o conjunto de dados
test_dataset = get_emnist_dataset(False)
print("dataset carregado com sucesso!")

# Criar o dataloader para separar os dados em lotes
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# modelo
# model = SimpleNeuralNetwork(args.input)
model = SimpleCNN()
file = checkpoint_path(args.input)
content = torch.load(file)
model.load_state_dict(content["model"])
class_names: list[str] = content["classes"]

model = model.to(device)

# desligando o calculo de gradientes
model.eval()
correct = 0
total = 0
# n: [total, correct]
number_map = {i: [0, 0] for i in range(len(class_names))}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label_val = int(labels[i].item())
            number_map[label_val][0] += 1
            if predicted[i].item() == label_val:
                number_map[label_val][1] += 1

acurracy = correct / total
print(f"Acurácia da rede de imagens de testes: {acurracy * 100:.2f}%")

classes = dict(sorted(number_map.items(), key=lambda kv: kv[1][1], reverse=True))

for i in classes:
    print(
        f"{class_names[i]}: {classes[i][1] / classes[i][0]:.4f}. [{classes[i][1]} de {classes[i][0]}]"
    )
