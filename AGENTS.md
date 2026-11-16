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

- `src\ocr\` is the runtime: `model.py` (CRNN, alphabet, shard dataset),
  `analyzer.py` (page to lines), `recognize.py` (page to text), `postprocess.py`
  (lexicon accent/case corrector).
- `src\training\` holds `train.py`, `evaluate.py`, and `compare_font_errors.py`.
- `src\datagen\` generates the synthetic dataset: `generator.py` writes the word
  and page-line shards, `page_generator.py` renders full pages with a JSON clue
  file.
- `src\web\` is the local debug UI: `app.py` (FastAPI, upload to lines to text)
  and `static\index.html`.
- `src\utils\paths.py` defines paths for data, checkpoints, and test images.
- Model checkpoints live in `artifacts\checkpoints\ocr\`; datasets and generated
  images live under `data\`.

## Data safety

- `data\test_images\` holds the 15 synthetic pages with their JSON clues. They are
  the regression set: never delete or regenerate over them.
- `src\datagen\generator.py` writes and overwrites shards under
  `data\raw\words-dataset\`. Regenerating the dataset discards the current one.
- Preserve existing user changes and data. Do not clean generated or data
  directories unless the task explicitly requires it.

## Commands

Run modules from the repository root and use the project interpreter, for example:

```powershell
.venv\Scripts\python.exe -m src.ocr.recognize 11263 meu-modelo
.venv\Scripts\python.exe -m src.training.evaluate --input meu-modelo
```

`recognize`, `train` and `evaluate` take the checkpoint name with no default
(the file lives in `artifacts\checkpoints\ocr\`, without `.pth`).

Start the debug UI (binds 127.0.0.1:8000, CPU inference). `OCR_CHECKPOINT` is
required:

```powershell
$env:OCR_CHECKPOINT="meu-modelo"; .venv\Scripts\python.exe -m src.web.app
```

Run the tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

Install dependencies only into the project environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
