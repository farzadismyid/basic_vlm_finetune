import time
import torch


def run_inference(
    model,
    processor,
    image,
    prompt,
    max_new_tokens=64,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    start_time = time.time()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    end_time = time.time()

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    inference_time = round(end_time - start_time, 2)

    return {
        "response": output_text[0],
        "time_seconds": inference_time,
    }
