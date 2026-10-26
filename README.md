# NUMBER OCR

Character recognition project using the EMNIST ByClass dataset
(digits, uppercase letters, and lowercase letters).

## Project Structure

- `src/neural_network/`: EMNIST training, evaluation, visualization, and desktop
  character recognition.
- `src/ocr/`: character-image preprocessing and the OCR demo pipeline.
- `src/search_block/`: synthetic labeled-image generation and character
  segmentation/validation.
- `src/utils/`: shared paths and project utilities.
- `data/raw/`: EMNIST dataset; torchvision stores its files in
  `data/raw/EMNIST/raw/`.
- `data/test_images/`: generated or sample images and their label files.
- `data/debug_errors/`: debug images produced by the OCR demo.
- `artifacts/checkpoints/`: trained model weights.

## Run

Run commands from the repository root using the project virtual environment:

```powershell
.venv\Scripts\python.exe -m src.neural_network.train
.venv\Scripts\python.exe -m src.neural_network.evaluate
.venv\Scripts\python.exe -m src.neural_network.visualize
.venv\Scripts\python.exe -m src.neural_network.desktop
```

Use `-i` or `--input` to select a checkpoint number, for example:

```powershell
.venv\Scripts\python.exe -m src.neural_network.train --input 0
```

To train a new ByClass model while reusing compatible layers from the
`emnist-3.0.pth`:

```powershell
.venv\Scripts\python.exe -m src.neural_network.train --input 3.0 --output 4.0
```

Existing checkpoints trained on EMNIST Balanced do not include the lowercase `i` class;
train and use the new checkpoint to recognize it. The ByClass dataset must be available
in `data/raw/`, and checkpoints are saved in `artifacts/checkpoints/`.

### Image generation and segmentation

Generate a sample image with character labels:

```powershell
.venv\Scripts\python.exe -m src.search_block.image_generator
```

Run the bounding-box validation examples:

```powershell
.venv\Scripts\python.exe -m src.search_block.validation
```

**Warning:** the validation script clears files in `data/test_images/` before
generating its examples. Only run it when that directory contains disposable files.

The OCR demo can be run with:

```powershell
.venv\Scripts\python.exe -m src.ocr.process
```

It expects the sample image configured in the script and clears files in
`data/debug_errors/` before writing debug output. Check those paths before running.

## Python environment

Run all Python and pip commands through the repository's `.venv` interpreter.
For example, install dependencies with:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
