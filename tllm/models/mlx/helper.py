import itertools
import math
import os
import re
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn

from tllm.commons.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.utils import get_model_path, get_weight_path
from tllm.schemas import SeqInput


def greedy_decode(logits: mx.array) -> List[int]:
    # logits shape: [seq_len, vocab_size]
    out = mx.argmax(logits, axis=-1)
    return out.tolist()  # TODO: first requests is too slow


def build_mlx_mask(q_len_list: List[int], k_len_list: List[int]) -> mx.array:
    mask_list = [
        mx.tril(mx.ones((L, S), dtype=mx.bool_), k=0) if L > 1 else mx.ones((L, S), dtype=mx.bool_)
        for (L, S) in zip(q_len_list, k_len_list)
    ]

    combined_mask = mx.zeros((sum(q_len_list), sum(k_len_list)), dtype=mx.bool_)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.shape[0], r_index : r_index + mask.shape[1]] = mask
        l_index += mask.shape[0]
        r_index += mask.shape[1]

    final_mask = mx.where(combined_mask, 0, -math.inf)
    return final_mask


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    q_len_list, k_len_list = request_cache.build(seq_input, cache_manager)

    return AttentionData(
        request_cache=request_cache,
        attn_mask=build_mlx_mask(q_len_list, k_len_list),
        uuid_list=seq_input.uuid_list,
    )


def quantization_func(config, model, state_dict):
    if getattr(config, "quantization", None) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in state_dict

        nn.quantize(
            model,
            **config.quantization,
            class_predicate=class_predicate,
        )
    else:
        model.set_dtype(mx.bfloat16)

    return model


def empty_func(h, mask, cache):
    # TODO
    return h


def read_from_safetensors(file_path: str, key_list: List[str]) -> Dict[str, "mx.array"]:
    tensors = {}
    weights = mx.load(file_path)
    for key in weights.keys():
        for prefix_key in key_list:
            if key.startswith(prefix_key):
                tensors[key] = weights[key]
    return tensors


def get_last_hidden_states(hidden_states: mx.array, seq_len_list: List[int]) -> mx.array:
    index_list = list(itertools.accumulate(seq_len_list[:-1]))
    seq_hidden_states = mx.split(hidden_states, index_list, axis=0)
    hidden_states = mx.concat([x[-1:, :] for x in seq_hidden_states], axis=0)
    return hidden_states


def merge_weight_func(
    layer_pattern: re.Pattern, name_list: str, cat_name_fmt: str, weights: Dict[str, mx.array]
) -> Dict[str, mx.array]:
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
            weights[name] = mx.concatenate([layer_weights[qkv] for qkv in name_list], axis=0)
            temp_w.pop(layer_idx)

    layer_idx_list = list(temp_w.keys())
    for layer_idx in layer_idx_list:
        if len(temp_w[layer_idx]) != 0:
            raise ValueError(
                f"merge [{cat_name_fmt}] failed, layer_idx: {layer_idx}, temp_w: {temp_w[layer_idx].keys()}"
            )
    return weights


def pop_weight_func(
    prefix_key_list: List[str], weights: Dict[str, mx.array], num_layers: int, start_idx: int, end_idx: int
) -> Dict[str, mx.array]:
    prefix_key_list += [f"model.layers.{i}." for i in range(num_layers) if not (start_idx <= i < end_idx)]
    key_list = list(weights.keys())
    for key in key_list:
        for prefix_key in prefix_key_list:
            if key.startswith(prefix_key):
                weights.pop(key)
    return weights


def default_merge_attn_weight(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.weight")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def default_merge_attn_bias(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.bias")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.bias"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def default_merge_mlp_weight(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*.weight")
    mlp_list = ["gate_proj", "up_proj"]
    mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
    return merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)


def read_state_dict(model_path: str) -> Dict[str, mx.array]:
    model_path = get_model_path(model_path)
    file_set, prefix_key_list = get_weight_path(model_path)
    state_dict = {}
    for file in file_set:
        weight_path = os.path.join(model_path, file)
        state_dict.update(read_from_safetensors(weight_path, prefix_key_list))
    return state_dict


def tie_embedding_weights(state_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
    has_key_list = list(state_dict.keys())
    if "lm_head.weight" not in state_dict:
        for key in has_key_list:
            if key.startswith("embed_tokens."):
                state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]
    return state_dict


def read_main_state_dict(state_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.") and not k.startswith("model.layers."):
            new_state_dict[k.split("model.")[-1]] = v
    state_dict = tie_embedding_weights(new_state_dict)
    return state_dict
