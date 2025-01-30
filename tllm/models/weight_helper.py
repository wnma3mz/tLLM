import re
from typing import Dict, List

from tllm import BACKEND, BackendEnum
from tllm.schemas import MIX_TENSOR

if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    from tllm.models.mlx.gguf_utils import load_gguf_weight
    from tllm.models.mlx.helper import read_from_safetensors

    cat_func = lambda tensors: mx.concat(tensors, axis=0)
elif BACKEND == BackendEnum.TORCH:
    import torch

    from tllm.models.torch.helper import read_from_safetensors

    cat_func = lambda tensors: torch.cat(tensors, dim=0)
    load_gguf_weight = lambda x: None, None, None


def pop_weight_func(
    prefix_key_list: List[str], weights: Dict[str, MIX_TENSOR], num_layers: int, start_idx: int, end_idx: int
) -> Dict[str, MIX_TENSOR]:
    prefix_key_list += [f"model.layers.{i}." for i in range(num_layers) if not (start_idx <= i < end_idx)]
    key_list = list(weights.keys())
    for key in key_list:
        for prefix_key in prefix_key_list:
            if key.startswith(prefix_key):
                weights.pop(key)
    return weights


def merge_weight_func(
    layer_pattern: re.Pattern, name_list: str, cat_name_fmt: str, weights: Dict[str, MIX_TENSOR]
) -> Dict[str, MIX_TENSOR]:
    key_list = list(weights.keys())

    temp_w = {}  # save merge weights
    merge_num = len(name_list)
    for key in key_list:
        res = layer_pattern.findall(key)
        if res:
            layer_idx = int(res[0])
            if layer_idx not in temp_w:
                temp_w[layer_idx] = {}
        else:
            continue

        for name in name_list:
            if name in key:
                temp_w[layer_idx].update({name: weights.pop(key)})

        layer_weights = temp_w.get(layer_idx, [])
        if len(layer_weights) == merge_num:
            name = cat_name_fmt.format(layer_idx=layer_idx)
            weights[name] = cat_func([layer_weights[qkv] for qkv in name_list])
            temp_w.pop(layer_idx)

    layer_idx_list = list(temp_w.keys())
    for layer_idx in layer_idx_list:
        if len(temp_w[layer_idx]) != 0:
            raise ValueError(
                f"merge [{cat_name_fmt}] failed, layer_idx: {layer_idx}, temp_w: {temp_w[layer_idx].keys()}"
            )
    return weights


def default_merge_attn(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_list = ["q_proj", "k_proj", "v_proj"]
    for suffix in ["weight", "biases", "scales", "bias"]:
        attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*\." + suffix + r"$")
        attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer." + suffix
        weights = merge_weight_func(attn_pattern, attn_list, attn_name, weights)
    return weights


def default_merge_mlp(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    mlp_list = ["gate_proj", "up_proj"]
    for suffix in ["weight", "biases", "scales", "bias"]:
        mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*." + suffix + r"$")
        mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer." + suffix
        weights = merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)
    return weights


def split_tensor_by_rank(tensor, world_size, rank, axis):
    """Split tensor along specified axis based on world size and return rank's portion."""
    indices = [i * tensor.shape[axis] // world_size for i in range(1, world_size)]
    return mx.split(tensor, indices, axis=axis)[rank]


def tensor_parallel_state_dict(state_dict, world_size, rank):
    """Process state dictionary by splitting specific layers across ranks."""
    attn_list = ["q_proj", "k_proj", "v_proj"]
    mlp_list = ["gate_proj", "up_proj"]

    suffix_list = ["weight", "biases", "scales", "bias"]

    axis_0_list = [f"self_attn.{layer_name}.{suffix}" for layer_name in attn_list for suffix in suffix_list]
    axis_0_list += [f"mlp.{layer_name}.{suffix}" for layer_name in mlp_list for suffix in suffix_list]

    axis_1_list = [f"self_attn.o_proj.layer.{suffix}" for suffix in suffix_list]
    axis_1_list += [f"mlp.down_proj.layer.{suffix}" for suffix in suffix_list]

    split_patterns = {"axis_0": axis_0_list, "axis_1": axis_1_list}

    processed_dict = state_dict.copy()

    for key in list(state_dict.keys()):
        if any(pattern in key for pattern in split_patterns["axis_0"]):
            processed_dict[key] = split_tensor_by_rank(state_dict[key], world_size, rank, axis=0)

        elif any(pattern in key for pattern in split_patterns["axis_1"]):
            processed_dict[key] = split_tensor_by_rank(state_dict[key], world_size, rank, axis=1)

    return processed_dict


def tie_word_embeddings_func(config, state_dict):
    if getattr(config, "tie_word_embeddings", False):
        key_list = list(filter(lambda x: "embed_tokens" in x, state_dict.keys()))
        for key in key_list:
            state_dict[key.replace("embed_tokens", "lm_head")] = state_dict[key]
    return state_dict
