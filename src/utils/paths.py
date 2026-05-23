from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================
# DATA
# =========================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAMPLES_DIR = DATA_DIR / "samples"

METADATA_DIR = SAMPLES_DIR / "metadata"

FASHION_IMAGES_DIR = SAMPLES_DIR / "fashion_images"


# =========================
# OUTPUTS
# =========================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

LOGS_DIR = OUTPUTS_DIR / "logs"


# =========================
# CREATE IMPORTANT DIRS
# =========================

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
