# NUMBER OCR

Projeto de reconhecimento de dígitos usando o conjunto de dados MNIST.

## Estrutura

- `src/`: módulos Python do projeto.
- `data/raw/`: conjunto de dados MNIST; o torchvision mantém os arquivos em `data/raw/MNIST/raw/`.
- `artifacts/checkpoints/`: pesos dos modelos treinados.

## Executar

Execute os comandos a partir da raiz do repositório:

```powershell
python -m src.neural_network.train
python -m src.neural_network.evaluate
python -m src.neural_network.visualize
python -m src.neural_network.desktop
```

Use `-i` ou `--input` para selecionar o índice do modelo, por exemplo:

```powershell
python -m src.neural_network.train --input 0
```

O MNIST será baixado para `data/raw/` quando necessário, e os checkpoints serão salvos em `artifacts/checkpoints/`.
