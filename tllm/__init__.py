from enum import Enum
import importlib.util
import os


class BackendEnum(Enum):
    TORCH = 1
    MLX = 2


if importlib.util.find_spec("mlx"):
    BACKEND = BackendEnum.MLX
elif importlib.util.find_spec("torch"):
    BACKEND = BackendEnum.TORCH
else:
    raise ImportError("No backend found")

if os.environ.get("TLLM_BACKEND", None):
    BACKEND = BackendEnum[os.environ["TLLM_BACKEND"]]

if BACKEND == BackendEnum.TORCH:
    import torch

    DTYPE = torch.float16
    CONVERT_DTYPE = torch.float16
    DES_DTYPE = torch.float16
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
        DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        DEVICE = "cpu"
else:
    import mlx.core as mx
    import numpy as np

    DTYPE = mx.bfloat16  # in mlx, float16 is slower than bfloat16
    CONVERT_DTYPE = mx.float16
    DES_DTYPE = np.float16
    DEVICE = None

GRPC_OPTIONS = [
    ("grpc.max_metadata_size", 32 * 1024 * 1024),
    ("grpc.max_send_message_length", 128 * 1024 * 1024),
    ("grpc.max_receive_message_length", 128 * 1024 * 1024),
]

PP_TIMEOUT = 10  # TTFT 不超过 10s，且单个 token 生成时间不超过 10s

MASTER_SOCKET_PATH = "/tmp/tllm_master_grpc_uds.sock"
CLIENT_SOCKET_PATH = "/tmp/tllm_client_grpc_uds.sock"
