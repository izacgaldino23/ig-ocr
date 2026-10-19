import torch
from torch.utils.data import DataLoader

from src.config import init_config

from .model import SimpleCNN, get_emnist_dataset
from .paths import checkpoint_path


def main():
    # 1. Configuração inicial
    args, device = init_config()

    # 2. Baixar e carregar o conjunto de dados
    train_dataset = get_emnist_dataset(True)
    print("2. dataset carregado com sucesso!")

    # 3. Criar o dataloader para separar os dados em lotes
    batch_size = 64
    workers = 1
    if device.type == "cuda":
        batch_size = 256
        workers = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=workers,
    )
    print(f"3. {len(train_loader)} lotes carregados com sucesso!")

    # 4. Criar a rede
    # model = SimpleNeuralNetwork(args.input)
    model = SimpleCNN()
    file = checkpoint_path(args.input)
    print("4. Rede criada com sucesso!")

    if file.exists():
        content = torch.load(file)
        model.load_state_dict(content["model"])
        print("5. Modelo carregado com sucesso!")
    else:
        print("5. Modulo novo!")

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Loop de treinamento
    epochs = 5
    print("6. Comecando o treinamento...")
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader, 0):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # zerar os gradientes
            optimizer.zero_grad()

            # previsao da rede
            outputs = model(images)

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
            correct += (predicted == labels).sum().item()

            if batch_idx % 200 == 199 or batch_idx == len(train_loader) - 1:
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

    print("7. Treinamento concluido!")

    if args.output != 0:
        file = checkpoint_path(args.output)

    file.parent.mkdir(parents=True, exist_ok=True)
    content = {"model": model.state_dict(), "classes": train_dataset.classes}
    torch.save(content, file)
    print(f"8. Modelo salvo com sucesso como {file}!")


if __name__ == "__main__":
    main()
