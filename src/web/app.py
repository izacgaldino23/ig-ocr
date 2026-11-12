"""Local debug UI for the OCR pipeline.

Upload one page, see the detected lines, then the recognized text per line.
Run from the repository root with `python -m src.web.app` and open
http://127.0.0.1:8000.
"""

import base64
import io
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from src.ocr.analyzer import Line, group_paragraphs, segment_page
from src.ocr.model import CHAR2IDX, CRNN
from src.ocr.postprocess import LexiconCorrector
from src.ocr.recognize import line_crop, page_text, recognize_line

CHECKPOINT = "0.4.8-ft"
DEVICE = torch.device("cpu")
MAX_WIDTH = 2500
MAX_UPLOAD = 20 * 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"

MODEL: CRNN | None = None
CORRECTOR: LexiconCorrector | None = None
INFERENCE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    model = CRNN(len(CHAR2IDX) + 1, CHECKPOINT)
    if not model.load_if_exists():
        raise RuntimeError(f"checkpoint {CHECKPOINT} not found")
    MODEL = model.to(DEVICE).eval()
    yield


app = FastAPI(title="ig-ocr debug", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_model() -> CRNN:
    if MODEL is None:
        raise HTTPException(503, "model not loaded")
    return MODEL


def get_corrector() -> LexiconCorrector:
    global CORRECTOR
    if CORRECTOR is None:
        CORRECTOR = LexiconCorrector()  # ~1s: loads the floresta corpus
    return CORRECTOR


def read_image(upload: UploadFile) -> np.ndarray:
    if upload.size and upload.size > MAX_UPLOAD:
        raise HTTPException(413, "imagem maior que 20 MB")

    data = upload.file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "imagem maior que 20 MB")

    try:
        # exif_transpose before any resize: phone photos are rotated by metadata
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "arquivo não é uma imagem legível") from None

    gray = np.array(image)

    # a 12MP photo is slow to segment and pointless to display at full size
    if gray.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / gray.shape[1]
        size = (MAX_WIDTH, max(1, round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    return gray


def png_data_uri(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "falha ao codificar PNG")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def draw_lines(image: np.ndarray, lines: list[Line]) -> str:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    width = image.shape[1]
    thickness = max(2, round(width / 600))
    font = max(0.5, width / 1600)
    color = (0, 140, 255)

    for idx, line in enumerate(lines, start=1):
        x, y, w, h = line.bbox.get_values()
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        # number sits in the left margin so it never covers the ink
        label_x = max(2, x - int(26 * font))
        cv2.putText(
            canvas,
            str(idx),
            (label_x, y + h // 2 + int(6 * font)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )

    return png_data_uri(canvas)


def draw_crop(image: np.ndarray, line: Line) -> str:
    roi = line_crop(image, line)
    return "" if roi.size == 0 else png_data_uri(roi)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
def analyze(
    file: Annotated[UploadFile, File()],
    dark_text: Annotated[bool | None, Form()] = None,
):
    image = read_image(file)

    # contours are found on white blobs, so dark text on light paper inverts
    inverted = bool(image.mean() > 127) if dark_text is None else dark_text

    start = time.perf_counter()
    lines = segment_page(image, inverted)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "dark_text_used": inverted,
        "lines": [
            {"id": idx, "bbox": line.bbox.to_dict()}
            for idx, line in enumerate(lines, start=1)
        ],
        "overlay_b64": draw_lines(image, lines),
        "ms": round(elapsed, 1),
    }


@app.post("/api/recognize")
def recognize(
    file: Annotated[UploadFile, File()],
    postprocess: Annotated[bool, Form()] = False,
    flow: Annotated[bool, Form()] = False,
    dark_text: Annotated[bool | None, Form()] = None,
):
    image = read_image(file)
    inverted = bool(image.mean() > 127) if dark_text is None else dark_text
    lines = segment_page(image, inverted)
    corrector = get_corrector() if postprocess else None
    model = get_model()
    paragraphs = group_paragraphs(lines)

    start = time.perf_counter()
    rows = []
    for idx, line in enumerate(lines, start=1):
        line_start = time.perf_counter()
        with INFERENCE_LOCK:
            text = recognize_line(model, DEVICE, image, line)
        line_ms = (time.perf_counter() - line_start) * 1000
        corrected = corrector.correct_line(text) if corrector else text

        rows.append(
            {
                "id": idx,
                "bbox": line.bbox.to_dict(),
                "text": corrected,
                "changed": corrected != text,
                "paragraph": paragraphs[idx - 1],
                "crop_b64": draw_crop(image, line),
                "ms": round(line_ms, 1),
            }
        )

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "lines": rows,
        "paragraphs": paragraphs[-1] + 1 if paragraphs else 0,
        "text": page_text([row["text"] for row in rows], paragraphs, flow),
        "ms": round(elapsed, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
