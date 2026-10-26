# NUMBER OCR

Character recognition project using the EMNIST ByClass dataset
(digits, uppercase letters, and lowercase letters).

## Project Structure

- `src/`: project Python modules.
- `data/raw/`: EMNIST dataset; torchvision stores its files in `data/raw/EMNIST/raw/`.
- `artifacts/checkpoints/`: trained model weights.

## Run

Run commands from the repository root:

```powershell
python -m src.neural_network.train
python -m src.neural_network.evaluate
python -m src.neural_network.visualize
python -m src.neural_network.desktop
```

Use `-i` or `--input` to select a model index, for example:

```powershell
python -m src.neural_network.train --input 0
```

To train a new ByClass model while reusing compatible layers from the
`emnist-3.0.pth`:

```powershell
python -m src.neural_network.train --input 3.0 --output 4.0
```

Existing checkpoints trained on EMNIST Balanced do not include the lowercase `i` class;
train and use the new checkpoint to recognize it. The ByClass dataset must be available
in `data/raw/`, and checkpoints are saved in `artifacts/checkpoints/`.
