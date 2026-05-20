import json
from datetime import datetime

from src.utils.paths import OUTPUTS_DIR


EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

EXPERIMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_experiment(results, experiment_name):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        EXPERIMENTS_DIR /
        f"{experiment_name}_{timestamp}.json"
    )

    with open(output_path, "w") as f:

        json.dump(results, f, indent=4)

    return output_path
