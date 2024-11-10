import json
import os
from typing import Dict, List, Set

from safetensors import safe_open
import torch

from tllm import HAS_MLX
from tllm.schemas import MIX_TENSOR, GenerateEnd


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


def read_from_safetensors(file_path: str, key_list: List[str]) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for prefix_key in key_list:
                if key.startswith(prefix_key):
                    tensors[key] = f.get_tensor(key)
    return tensors


def read_from_mlx_safetensors(file_path: str, key_list: List[str]) -> Dict[str, "mx.array"]:
    import mlx.core as mx

    tensors = {}
    weights = mx.load(file_path)
    for key in weights.keys():
        for prefix_key in key_list:
            if key.startswith(prefix_key):
                tensors[key] = weights[key]
    return tensors


def load_master_weight(model_path: str) -> Dict[str, MIX_TENSOR]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    file_set = set()
    prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        for key, file_ in index["weight_map"].items():
            for prefix_key in prefix_key_list:
                if key.startswith(prefix_key):
                    file_set.add(file_)
    else:
        file_set.add("model.safetensors")

    weight_dict = {}
    for file in file_set:
        weight_path = os.path.join(model_path, file)
        if HAS_MLX:
            weight_dict.update(read_from_mlx_safetensors(weight_path, prefix_key_list))
        else:
            weight_dict.update(read_from_safetensors(weight_path, prefix_key_list))
    return weight_dict
