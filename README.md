# NUMBER OCR

Projeto de reconhecimento de caracteres usando o conjunto de dados EMNIST ByClass
(dígitos, letras maiúsculas e minúsculas).

## Estrutura

- `src/`: módulos Python do projeto.
- `data/raw/`: conjunto de dados EMNIST; o torchvision mantém os arquivos em `data/raw/EMNIST/raw/`.
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

Para treinar um novo modelo ByClass reaproveitando as camadas compatíveis do checkpoint
`emnist-3.0.pth`:

```powershell
python -m src.neural_network.train --input 3.0 --output 4.0
```

Os checkpoints antigos treinados com EMNIST Balanced não incluem a classe `i` minúscula;
é necessário treinar e usar o novo checkpoint para reconhecê-la. O conjunto ByClass deve
estar disponível em `data/raw/`, e os checkpoints são salvos em `artifacts/checkpoints/`.
