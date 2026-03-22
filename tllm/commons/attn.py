from tllm import BACKEND, BackendEnum

if BACKEND == BackendEnum.MLX:
    ATTN_FUNC, ATTN_TYPE = None, "mlx"
else:
    raise ValueError("Only MLX backend is supported")
