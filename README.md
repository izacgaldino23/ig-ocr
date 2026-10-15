# NUMBER OCR

Projeto de reconhecimento de dígitos usando o conjunto de dados MNIST.

## Estrutura

- `src/`: módulos Python do projeto.
- `data/raw/`: conjunto de dados MNIST; o torchvision mantém os arquivos em `data/raw/MNIST/raw/`.
- `artifacts/checkpoints/`: pesos dos modelos treinados.

## Executar

Execute os comandos a partir da raiz do repositório:

```powershell
python -m src.train
python -m src.evaluate
python -m src.visualize
python -m src.desktop
```

Use `-i` ou `--input` para selecionar o índice do modelo, por exemplo:

```powershell
python -m src.train --input 0
```

O MNIST será baixado para `data/raw/` quando necessário, e os checkpoints serão salvos em `artifacts/checkpoints/`.

## Modelos testados

### Modelo 0

- Entrada: 28 × 28
- Camadas ocultas: 128 e 64 neurônios, com ReLU
- Saída: 10 classes