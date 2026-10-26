# Repository Guidance

## Python environment

- Run every Python command from the repository root with `.venv\Scripts\python.exe`.
- Run pip through that interpreter, for example:
  `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Do not use system `python`, `py`, or `pip`, and do not rely on shell activation
  to select the interpreter. This keeps commands, tests, and installs inside the
  project environment.
- Use the project's `.venv\Scripts\ruff.exe` for linting when needed.

## Project layout

- `src\neural_network\` contains EMNIST ByClass training, evaluation, visualization,
  and the desktop recognizer.
- `src\ocr\` contains character ROI preprocessing and the OCR demo pipeline.
- `src\search_block\` generates labeled sample images and detects/validates
  character bounding boxes.
- `src\utils\paths.py` defines paths for data, checkpoints, test images, and debug
  output.
- Model checkpoints are stored in `artifacts\checkpoints\`; datasets and generated
  images are stored under `data\`.

## Data safety

- `.venv\Scripts\python.exe -m src.search_block.validation` deletes files in
  `data\test_images\` before generating its sample images. Only run it when that
  directory contains disposable files.
- `.venv\Scripts\python.exe -m src.ocr.process` clears files in
  `data\debug_errors\` and expects its sample image to be available. Check those
  paths before running it.
- Preserve existing user changes and data. Do not clean generated or data
  directories unless the task explicitly requires it.

## Commands

Run modules from the repository root and use the project interpreter, for example:

```powershell
.venv\Scripts\python.exe -m src.neural_network.evaluate
```

Install dependencies only into the project environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```