from datetime import datetime


def create_experiment_id():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
