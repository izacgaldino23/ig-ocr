import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleNeuralNetwork
from args import get_args

args = get_args()

# 1. Transformação para converter as imagens em tensores e normaliza-las
transform = transforms.Compose([
    transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomErasing(p=0.1, value='random'),
])
print("1. transformers carregados com sucesso!")

# 2. Baixar e carregar o conjunto de dados
train_dataset = datasets.MNIST('./data', train=True, transform=transform)
print("2. dataset carregado com sucesso!")

# 3. Criar o dataloader para separar os dados em lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print(f"3. {len(train_loader)} lotes carregados com sucesso!")

# 4. Criar a rede
model = SimpleNeuralNetwork(args.input)
file = f'data/models/mnist-{args.input}.pth'
print("4. Rede criada com sucesso!")

if os.path.exists(file):
    model.load_state_dict(torch.load(file))
    print("5. Modelo carregado com sucesso!")
else:
    print("5. Modulo novo!")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loop de treinamento
epochs = 3
print("6. Comecando o treinamento...")
total = 0
correct = 0
for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (image, labels) in enumerate(train_loader, 0):
        
        # zerar os gradientes
        optimizer.zero_grad()
        
        # previsao da rede
        outputs = model(image)
        
        # calculo de erro
        loss = loss_fn(outputs, labels)
        
        # retropropagação
        loss.backward()
        
        # atualiza os pesos
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 200 == 199 or batch_idx == len(train_loader)-1: # Imprime a cada 200 lotes e no final
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            print(f"\t=> Época [{epoch+1}/{epochs}], Lote [{batch_idx+1}/{len(train_loader)}], Perda: {running_loss / 200:.4f}", end=' ')
            print(f"\r\t=Acurácia: {correct / total:.4f}", end='')
            running_loss = 0.0
            
print("7. Treinamento concluido!")

torch.save(model.state_dict(), file)
print(f"8. Modelo salvo com sucesso como {file}!")