import glob
import os
from typing import Dict

from transformers import LlamaForCausalLM

from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyLlamaModel)}

try:
    import mlx.core as mx  # type: ignore

    HAS_MLX = True
except:
    HAS_MLX = False

if HAS_MLX:
    from tllm.models.mlx_llama import MyMLXLlamaModel

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyMLXLlamaModel)})

    def load_weight(model_path: str) -> Dict[str, mx.array]:
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        return weights

else:

    def load_weight(model_path: str) -> Dict[str, "mx.array"]:
        return {}
