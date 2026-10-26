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
.venv\Scripts\python.exe -m src.neural_network.train --input 0 --output 1
```

For a worker-count comparison, always start from the same input checkpoint and
write each run to a different output checkpoint so the source checkpoint stays
unchanged. For example:

```powershell
.venv\Scripts\python.exe -m src.neural_network.train --input 0 --output 1 --workers 2 --seed 42
.venv\Scripts\python.exe -m src.neural_network.train --input 0 --output 2 --workers 4 --seed 42
```

Use `--workers 0` to compare single-process loading, or `--batch-size` to test
another batch size. The defaults remain 4 workers and batch size 256 on CUDA.
Evaluation uses fixed orientation/normalization preprocessing (no random
augmentation), so repeated evaluations of the same checkpoint are comparable.

To try a slightly smaller CNN while initializing it from the existing model,
choose a new output checkpoint and add `--compact`:

```powershell
.venv\Scripts\python.exe -m src.neural_network.train --input 0 --output 3 --workers 2 --seed 42 --compact
.venv\Scripts\python.exe -m src.neural_network.evaluate --input 3
```

The compact variant uses 28/56/112/112 convolution channels instead of
32/64/128/128. Compatible convolutional, normalization, and dense weights are
sliced from the input checkpoint to initialize it. The checkpoint records its
architecture so evaluation and recognition tools can load either model.

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
