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


def default_merge_attn_bias(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.bias")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.bias"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def default_merge_mlp_weight(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*.weight")
    mlp_list = ["gate_proj", "up_proj"]
    mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
    return merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)


def default_merge_attn_weight(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.weight")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def default_merge_mlp_quantization(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    mlp_list = ["gate_proj", "up_proj"]
    mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*.biases")
    mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer.biases"
    weights = merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)

    mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*.scales")
    mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer.scales"
    return merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)


def default_merge_attn_quantization(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.biases")
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.biases"
    weights = merge_weight_func(attn_pattern, attn_list, attn_name, weights)

    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.scales")
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.scales"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def tie_embedding_weights(state_dict: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    has_key_list = list(state_dict.keys())
    if "lm_head.weight" not in state_dict:
        for key in has_key_list:
            if key.startswith("embed_tokens."):
                state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]
    return state_dict
