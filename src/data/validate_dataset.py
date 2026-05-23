def validate_sample(sample):

    required_keys = [
        "image_name",
        "messages",
    ]

    for key in required_keys:

        if key not in sample:

            return False

    return True


def validate_dataset(dataset):

    valid_samples = []

    invalid_samples = []

    for sample in dataset:

        if validate_sample(sample):

            valid_samples.append(sample)

        else:

            invalid_samples.append(sample)

    return {
        "valid_count": len(valid_samples),
        "invalid_count": len(invalid_samples),
    }
