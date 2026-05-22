from statistics import mean


def summarize_results(results):

    times = [
        result["time_seconds"]
        for result in results
    ]

    response_lengths = [
        len(result["response"].split())
        for result in results
    ]

    summary = {

        "total_samples":
            len(results),

        "average_inference_time":
            round(mean(times), 2),

        "average_response_length":
            round(mean(response_lengths), 2),
    }

    return summary
