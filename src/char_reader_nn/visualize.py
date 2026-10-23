import matplotlib.pyplot as plt
import torch

from src.char_reader_nn.config import init_config
from src.char_reader_nn.model import MEAN, STD, SimpleCNN, get_emnist_dataset
from src.utils.paths import checkpoint_path

# 1. Configuração inicial
args, device = init_config(True)

# 2. Baixar e carregar o conjunto de dados
test_dataset = get_emnist_dataset(False)

# 3. modelo
# model = SimpleNeuralNetwork(args.input)
model = SimpleCNN()
file = checkpoint_path(args.input)
content = torch.load(file)
model.load_state_dict(content["model"])
class_names = content["classes"]

model = model.to(device)

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

        predict_idx = int(predicted.item())
        predicted_item = class_names[predict_idx]

        real_label = class_names[label]
        if predict_idx != label:
            # Prepara a imagem para exibição (removendo a normalização e convertendo para numpy)
            img_display = image.squeeze(0) * STD + MEAN

            axes[wrong].imshow(img_display, cmap="gray")
            axes[wrong].set_title(f"Real: {real_label}\nPred: {predicted_item}")
            print(f"Real: {real_label}\t Pred: {predicted_item}\t Index: {i}")
            axes[wrong].axis("off")

            wrong += 1

        if wrong == 5 or i == len(test_dataset) - 1:
            break
        i += 1

plt.tight_layout()
plt.show()
