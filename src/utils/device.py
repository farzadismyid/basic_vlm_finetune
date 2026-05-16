import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def print_device_info():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
