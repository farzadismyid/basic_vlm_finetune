from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
