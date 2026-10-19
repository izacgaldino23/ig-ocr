import torch
from torch.utils.data import DataLoader

from .args import get_args
from .model import SimpleCNN, get_emnist_dataset, get_transformers
from .paths import checkpoint_path

args = get_args()

# 1. Transformação para converter as imagens em tensores e normaliza-las
transformers = get_transformers()
print("1. transformers carregados com sucesso!")

# 2. Baixar e carregar o conjunto de dados
train_dataset = get_emnist_dataset(True, False)
class_names = train_dataset.classes
print("2. dataset carregado com sucesso!")

# 3. Criar o dataloader para separar os dados em lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print(f"3. {len(train_loader)} lotes carregados com sucesso!")

# 4. Criar a rede
# model = SimpleNeuralNetwork(args.input)
model = SimpleCNN()
file = checkpoint_path(args.input)
print("4. Rede criada com sucesso!")

if file.exists():
    model.load_state_dict(torch.load(file))
    print("5. Modelo carregado com sucesso!")
else:
    print("5. Modulo novo!")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loop de treinamento
epochs = 5
print("6. Comecando o treinamento...")
for epoch in range(epochs):
    running_loss = 0.0
    total = 0
    correct = 0
    batches_interval = 0
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

        running_loss += float(loss.item())

        # Acurácia
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        predicted_idx = predicted[0].item()
        correct += (predicted == labels).sum().item()

        if (batch_idx % 200 == 199 or batch_idx == len(train_loader) - 1):
            avg_loss = running_loss / batch_idx
            accuracy = correct / total
            print(
                f"\t=> Época [{epoch + 1}/{epochs}], "
                f"Lote [{batch_idx + 1}/{len(train_loader)}], "
                f"Perda: {avg_loss:.4f}, "
                f"Acurácia: {accuracy * 100:.2f}%"
            )
            # Reseta o acumulador da perda para o próximo intervalo
            running_loss = 0.0
            batches_in_interval = 0

print("7. Treinamento concluido!")

if args.output != 0:
    file = checkpoint_path(args.output)

file.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), file)
print(f"8. Modelo salvo com sucesso como {file}!")
