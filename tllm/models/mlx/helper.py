from dataclasses import make_dataclass
import itertools
import math
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn

from tllm import DTYPE, ENABLE_PREFILL_CACHE
from tllm.commons.cache import AttentionData
from tllm.commons.cache_manager import CacheManager
from tllm.schemas import SeqInput


def greedy_decode(logits: mx.array) -> List[int]:
    # logits shape: [seq_len, vocab_size]
    out = mx.argmax(logits, axis=-1)
    return out.tolist()  # TODO: first requests is too slow


def build_mlx_mask(q_len_list: List[int], k_len_list: List[int], hit_cache_len_list: List[int]) -> mx.array:
    mask_list = []
    sum_q_len = sum(q_len_list)
    sum_k_len = sum(k_len_list)
    for q_len, k_len, hit_cache_len in zip(q_len_list, k_len_list, hit_cache_len_list):
        # prefilling
        if q_len > 1:
            mask = mx.tril(mx.ones((q_len, k_len), dtype=mx.bool_), k=0)
            if hit_cache_len != -1:
                sum_q_len -= hit_cache_len
                mask = mask[-(q_len - hit_cache_len) :]
        else:
            mask = mx.ones((q_len, k_len), dtype=mx.bool_)
        mask_list.append(mask)

    combined_mask = mx.zeros((sum_q_len, sum_k_len), dtype=mx.bool_)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.shape[0], r_index : r_index + mask.shape[1]] = mask
        l_index += mask.shape[0]
        r_index += mask.shape[1]

    final_mask = mx.where(combined_mask, 0, -math.inf).astype(DTYPE)
    return final_mask


class MLXCacheManager(CacheManager):
    def build_forward_cache(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        q_len_list, k_len_list, position_ids_list, hit_cache_len_list = self.build_cache(seq_input, self.cache)

        self.hit_cache_flag = any(x != -1 for x in hit_cache_len_list)

        # 截断 hidden_states
        if ENABLE_PREFILL_CACHE and self.is_start_pp and self.hit_cache_flag:
            hidden_states_list = []
            q_start = 0
            for q_len, hit_cache_len in zip(q_len_list, hit_cache_len_list):
                if hit_cache_len != -1:
                    hidden_states_list.append(hidden_states[q_start : q_start + q_len][hit_cache_len:])
                else:
                    hidden_states_list.append(hidden_states[q_start : q_start + q_len])
                q_start += q_len
            hidden_states = mx.concat(hidden_states_list, axis=0)

        if hidden_states.dtype == mx.float16:  # float16 is much slower than bfloat16
            hidden_states = hidden_states.astype(mx.bfloat16)

        attn_mask = build_mlx_mask(q_len_list, k_len_list, hit_cache_len_list)

        self.q_len_list = q_len_list
        self.hit_cache_len_list = hit_cache_len_list
        self.attn_data = AttentionData(
            request_cache=self.request_cache,
            attn_mask=attn_mask if attn_mask is None else attn_mask.astype(hidden_states.dtype),
            uuid_list=seq_input.uuid_list,
        )

        self.position_ids = mx.concatenate(position_ids_list, axis=-1)

        return hidden_states

    def get_last_hidden_states(self, hidden_states: mx.array) -> mx.array:
        split_len_list = self.q_len_list
        if self.hit_cache_flag:
            q_start = 0
            for i, (q_len, hit_cache_len) in enumerate(zip(self.q_len_list, self.hit_cache_len_list)):
                if hit_cache_len != -1:
                    split_len_list[i] = q_len - hit_cache_len
                q_start += q_len
        if self.is_end_pp:
            index_list = list(itertools.accumulate(split_len_list[:-1]))
            seq_hidden_states = mx.split(hidden_states, index_list, axis=0)
            hidden_states = mx.concat([x[-1:, :] for x in seq_hidden_states], axis=0)
        return hidden_states


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


def read_from_safetensors(file_path: str) -> Dict[str, mx.array]:
    return mx.load(file_path)


def dict_to_dataclass(data: dict, name: str):
    """将字典转换为 dataclass

    Args:
        data: 字典数据
        name: dataclass 名称

    Returns:
        dataclass 对象
    """
    fields = [(key, type(value)) for key, value in data.items()]
    DataClass = make_dataclass(name, fields)
    return DataClass(**data)
