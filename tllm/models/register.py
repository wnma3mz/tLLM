# coding: utf-8
from tllm import BACKEND, BackendEnum

MODEL_REGISTER = {}
DEP_MODEL_REGISTER = {}


if BackendEnum.MLX == BACKEND:
    from tllm.models.qwen3_5 import MLXQwen35ForConditionalGeneration, MLXQwen35Model
    MODEL_REGISTER.update({"Qwen3_5ForConditionalGeneration": (MLXQwen35ForConditionalGeneration, MLXQwen35Model)})

    from tllm.models.backend_mlx.helper import greedy_decode

    sampling_func = greedy_decode
else:
    raise ValueError(f"Only MLX backend with Qwen3.5 is supported now, got backend={BACKEND}.")
