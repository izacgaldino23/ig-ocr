import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
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

# 3. modelo
model = SimpleNeuralNetwork(args.input)
file = f'data/models/mnist-{args.input}.pth'
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
            img_display = image.squeeze(0) * 0.3081 + 0.1307
            
            axes[wrong].imshow(img_display, cmap='gray')
            axes[wrong].set_title(f"Real: {label}\nPred: {predicted.item()}")
            print(f"Real: {label}\tPred: {predicted.item()}\tIndex: {i}")
            axes[wrong].axis('off')
            
            wrong += 1

        if wrong == 5 or i == len(test_dataset) - 1:
            break
        i += 1
        
plt.tight_layout()
plt.show()