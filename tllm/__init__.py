from enum import Enum
import importlib.util


class BackendEnum(Enum):
    TORCH = 1
    MLX = 2


if importlib.util.find_spec("mlx"):
    BACKEND = BackendEnum.MLX
elif importlib.util.find_spec("torch"):
    BACKEND = BackendEnum.TORCH


else:
    raise ImportError("No backend found")

if BACKEND == BackendEnum.TORCH:
    import torch

    DTYPE = torch.bfloat16
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
        DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        DEVICE = "cpu"
else:
    import mlx.core as mx

    DTYPE = mx.float16
    DEVICE = None
