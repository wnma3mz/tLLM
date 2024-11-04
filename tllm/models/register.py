import glob
import os
from typing import Dict

from transformers import LlamaForCausalLM

from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyLlamaModel)}

from tllm import HAS_MLX

if HAS_MLX:
    import mlx.core as mx  # type: ignore

    from tllm.models.mlx_llama import MyMLXLlamaForCausalLM, MyMLXLlamaModel

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (LlamaForCausalLM, MyMLXLlamaForCausalLM, MyMLXLlamaModel)})

    def load_weight(model_path: str) -> Dict[str, mx.array]:
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        return weights

else:

    def load_weight(model_path: str) -> Dict[str, "mx.array"]:
        return {}
