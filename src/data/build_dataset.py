import json

from pathlib import Path

from src.data.dataset_schema import (
    create_training_sample,
)


def build_dataset(manifest):

    dataset = []

    for item in manifest:

        sample = create_training_sample(

            image_name=item["image_name"],

            instruction=(
                "Describe this fashion outfit professionally."
            ),

            response=(
                f"A {item['style']} "
                f"{item['category']} outfit."
            ),
        )

        dataset.append(sample)

    return dataset


def save_dataset(dataset, output_path):

    output_path = Path(output_path)

    with open(output_path, "w") as f:

        json.dump(dataset, f, indent=4)
