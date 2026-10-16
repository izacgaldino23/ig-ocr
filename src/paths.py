from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"


def checkpoint_path(model_number: float) -> Path:
    return CHECKPOINT_DIR / f"mnist-{model_number}.pth"
