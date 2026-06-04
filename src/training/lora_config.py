from peft import LoraConfig


def get_lora_config():

    return LoraConfig(

        r=16,

        lora_alpha=32,

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",
    )
