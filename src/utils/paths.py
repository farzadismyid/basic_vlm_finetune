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

EXTERNAL_DATA_DIR = DATA_DIR / "external"

POLYVORE_DIR = EXTERNAL_DATA_DIR / "polyvore"

POLYVORE_RAW_DIR = POLYVORE_DIR / "raw"

POLYVORE_PROCESSED_DIR = (
    POLYVORE_DIR / "processed"
)

POLYVORE_METADATA_DIR = (
    POLYVORE_DIR / "metadata"
)

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
