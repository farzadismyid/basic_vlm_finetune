import json
from datetime import datetime

from src.utils.paths import PREDICTIONS_DIR


def save_prediction(data):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        PREDICTIONS_DIR /
        f"prediction_{timestamp}.json"
    )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    return output_path
