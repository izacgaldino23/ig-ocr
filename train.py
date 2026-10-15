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
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# 3. Criar o dataloader para separar os dados em lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Todos os dados foram carregados com sucesso! Total de {len(train_loader)} lotes de treino.")

# 4. Criar a rede
model = SimpleNeuralNetwork()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loop de treinamento
epochs = 3
print("Comecando o treinamento...")
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
            print(f"Época [{epoch+1}/{epochs}], Lote [{batch_idx+1}/{len(train_loader)}], Perda: {running_loss / 200:.4f}")
            running_loss = 0.0
            
print("Treinamento concluido!")

torch.save(model.state_dict(), 'mnist.pth')
print("Modelo salvo com sucesso como mnist.pth!")