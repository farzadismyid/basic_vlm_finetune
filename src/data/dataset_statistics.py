from statistics import mean


def dataset_statistics(dataset):

    response_lengths = []

    for sample in dataset:

        response = sample["messages"][1]["content"]

        response_lengths.append(
            len(response.split())
        )

    stats = {

        "total_samples":
            len(dataset),

        "average_response_length":
            round(mean(response_lengths), 2),
    }

    return stats
