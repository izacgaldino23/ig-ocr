from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test_images"


def checkpoint_path(model_number, output_folder="") -> Path:
    folder = CHECKPOINT_DIR if output_folder == "" else CHECKPOINT_DIR / output_folder
    return folder / f"{model_number}.pth"


def test_image_path(file_name: str) -> Path:
    return TEST_IMAGE_DIR / f"{file_name}"
