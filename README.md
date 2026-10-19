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

### Modelo 1

- Entrada: 28 × 28
- Camadas ocultas:
	- Convolucionais (densas): 
		- 1ª camada: 16 filtros de 3x3, ReLU e Pooling 2x2
		- 2ª camada: 32 filtros de 3x3, ReLU e Pooling 2x2
	- Linear:
		- Flatten
		- 64 neurônios com ReLU
- Saída: 10 classes

#### Modelo 1.1

Foi melhorado os transformers para ter algumas variações melhores usando um randomErasing personalizado.

### Modelo 2

Inicio da leitura do EMINIST (adicionar previsão de letras)

- Entrada: 28 × 28
- Camadas ocultas:
	- Convolucionais (densas): 
		- 1ª camada: 16 filtros de 3x3, ReLU e Pooling 2x2
		- 2ª camada: 32 filtros de 3x3, ReLU e Pooling 2x2
	- Linear:
		- Flatten
		- 64 neurônios com ReLU
- Saída: 47 classes
