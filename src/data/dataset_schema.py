def create_training_sample(
    image_name,
    instruction,
    response,
):

    sample = {

        "image_name": image_name,

        "messages": [

            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_name,
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            },

            {
                "role": "assistant",
                "content": response,
            },
        ],
    }

    return sample
