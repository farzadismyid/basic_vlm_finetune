import torch

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)


from src.config.settings import MODEL_ID

def load_qwen2vl():

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    return model, processor
