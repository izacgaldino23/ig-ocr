# NUMBER OCR

Projeto de reconhecimento de dígitos usando o conjunto de dados MNIST.

## Estrutura

- `src/`: módulos Python do projeto.
- `data/raw/`: conjunto de dados MNIST; o torchvision mantém os arquivos em `data/raw/MNIST/raw/`.
- `artifacts/checkpoints/`: pesos dos modelos treinados.

## Executar

Execute os comandos a partir da raiz do repositório:

```powershell
python -m src.char_reader_nn.train
python -m src.char_reader_nn.evaluate
python -m src.char_reader_nn.visualize
python -m src.char_reader_nn.desktop
```

Use `-i` ou `--input` para selecionar o índice do modelo, por exemplo:

```powershell
python -m src.char_reader_nn.train --input 0
```

O MNIST será baixado para `data/raw/` quando necessário, e os checkpoints serão salvos em `artifacts/checkpoints/`.
