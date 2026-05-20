from src.models.inference import run_inference


def run_batch_inference(
    model,
    processor,
    image_paths,
    image_loader,
    prompt,
):

    results = []

    for image_path in image_paths:

        image = image_loader(image_path)

        output = run_inference(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
        )

        results.append(
            {
                "image_path": str(image_path),
                "response": output["response"],
                "time_seconds": output["time_seconds"],
            }
        )

    return results
