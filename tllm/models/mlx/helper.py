import itertools
import math
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn

from tllm import DTYPE
from tllm.commons.cache import AttentionData, CacheManager, RequestsCache
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

    final_mask = mx.where(combined_mask, 0, -math.inf).astype(DTYPE)
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
        model.set_dtype(DTYPE)

    return model


def read_from_safetensors(file_path: str, key_list: List[str]) -> Dict[str, mx.array]:
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
