from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

SAMPLES_DIR = DATA_DIR / "samples"

METADATA_DIR = SAMPLES_DIR / "metadata"

FASHION_IMAGES_DIR = SAMPLES_DIR / "fashion_images"


OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
