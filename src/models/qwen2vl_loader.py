import torch

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def load_qwen2vl():

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor
