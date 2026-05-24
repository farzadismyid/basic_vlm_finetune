import torch


def gpu_memory_summary():

    if not torch.cuda.is_available():

        return {
            "cuda_available": False
        }

    allocated = (
        torch.cuda.memory_allocated() / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved() / 1024**3
    )

    return {

        "cuda_available": True,

        "allocated_gb":
            round(allocated, 2),

        "reserved_gb":
            round(reserved, 2),
    }
