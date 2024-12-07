from enum import Enum
import importlib


class BackendEnum(Enum):
    TORCH = 1
    MLX = 2


if importlib.util.find_spec("mlx"):
    BACKEND = BackendEnum.MLX
elif importlib.util.find_spec("torch"):
    BACKEND = BackendEnum.TORCH
else:
    raise ImportError("No backend found")
