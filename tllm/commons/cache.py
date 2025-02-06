# coding: utf-8
import copy
from dataclasses import dataclass
import time
from typing import Dict, Generic, List, Optional, TypeVar

from tllm import BACKEND, DEVICE, DTYPE, ENABLE_PREFILL_CACHE, BackendEnum
from tllm.commons.radix_tree import RadixTree
from tllm.schemas import MIX_TENSOR, SeqInput

if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    cat_func = lambda tensors: mx.concat(tensors, axis=0)
    zeros_func = lambda x0, x1, x2: mx.zeros(shape=(x0, x1, x2), dtype=DTYPE)
    array_func = lambda x: mx.array([x], dtype=mx.int32)
    arange_func = lambda x: mx.arange(0, x, dtype=mx.int32)
else:
    import torch

    cat_func = lambda tensors: torch.cat(tensors, dim=0)
    zeros_func = lambda x0, x1, x2: torch.zeros(size=(x0, x1, x2), dtype=DTYPE, device=DEVICE)
    array_func = lambda x: torch.tensor([x], dtype=torch.long)
    arange_func = lambda x: torch.arange(0, x, dtype=torch.long)


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    timestamp: float


class Cache(Generic[T]):
    def __init__(self, max_alive_time: int = 60):
        self._cache: Dict[str, CacheEntry[T]] = {}
        self.max_alive_time = max_alive_time

    def get(self, key: str) -> Optional[T]:
        if not self.contains(key):
            return None
        entry = self._cache[key]
        entry.timestamp = time.time()  # 更新访问时间
        return entry.value

    def set(self, key: str, value: T) -> None:
        self._cache[key] = CacheEntry(value=value, timestamp=time.time())

    def contains(self, key: str) -> bool:
        return key in self._cache

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    def check_alive(self) -> None:
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items() if current_time - entry.timestamp > self.max_alive_time
        ]
        for key in expired_keys:
            self.delete(key)


class KVCache:
    def __init__(
        self, max_seq_len: Optional[int] = -1, num_key_value_heads: Optional[int] = -1, head_dim: Optional[int] = -1
    ) -> None:
        # key_states/value_states: seq_len x num_heads x head_dim
        if max_seq_len == -1:
            self.key_states: Optional[MIX_TENSOR] = None
            self.value_states: Optional[MIX_TENSOR] = None
        else:
            self.key_states = zeros_func(max_seq_len, num_key_value_heads, head_dim)
            self.value_states = zeros_func(max_seq_len, num_key_value_heads, head_dim)
        self.kv_len = 0

    def set_kv_len(self, kv_len: int):
        self.kv_len = kv_len


class DecoderCache:
    def __init__(
        self, num_layers: int, q_len: int, max_seq_len: int = -1, num_key_value_heads: int = -1, head_dim: int = -1
    ):
        self.kv_cache_list = [KVCache(max_seq_len, num_key_value_heads, head_dim) for _ in range(num_layers)]
        self.q_len = q_len

    def __getitem__(self, idx: int) -> KVCache:
        return self.kv_cache_list[idx]

    def truncate(self, len_: int):
        for kv_cache in self.kv_cache_list:
            kv_cache.key_states = kv_cache.key_states[:len_]
            kv_cache.value_states = kv_cache.value_states[:len_]
            kv_cache.set_kv_len(len_)

    def set_q_len(self, q_len: int):
        self.q_len = q_len


class RequestsCache:
    def __init__(
        self, num_layers: int, max_seq_len: int = -1, num_key_value_heads: int = -1, head_dim: int = -1
    ) -> None:
        self.cache_dict: Dict[str, DecoderCache] = {}
        self.num_layers = num_layers
        self.max_seq_len, self.num_key_value_heads, self.head_dim = max_seq_len, num_key_value_heads, head_dim
        if self.max_seq_len == -1:
            # not cat attention to save memory
            self.update = self.update_cat
        else:
            # cat attention to save time
            self.update = self.update_no_cat
        self.radix_tree = RadixTree()

    def clear(self):
        self.cache_dict.clear()

    def insert_cache(self, seq_input: SeqInput):
        if ENABLE_PREFILL_CACHE:
            for input_ids, request_id in zip(seq_input.input_ids_list, seq_input.uuid_list):
                self.radix_tree.append_to_request(input_ids, request_id)

    def add(self, uuid: str, q_len: int, decoder_cache: Optional[DecoderCache] = None):
        # 保存每个 uuid 请求所有层的 cache
        self.cache_dict[uuid] = (
            DecoderCache(self.num_layers, q_len, self.max_seq_len, self.num_key_value_heads, self.head_dim)
            if decoder_cache is None
            else decoder_cache
        )

    def get_decoder_cache(self, uuid: str) -> DecoderCache:
        return self.cache_dict[uuid]

    def get_layer_idx_kv_cache(self, uuid: str, layer_idx: int) -> KVCache:
        return self.get_decoder_cache(uuid)[layer_idx]

    def get_q_len(self, uuid: str) -> int:
        # 获取每个 uuid 请求的 q_len
        return self.get_decoder_cache(uuid).q_len

    def get_kv_len(self, uuid: str, layer_idx: Optional[int] = 0) -> int:
        # 获取每个 uuid 请求的 kv cache 的 kv_len
        return self.get_layer_idx_kv_cache(uuid, layer_idx).kv_len

    def get_offset_list(self, uuid_list: List[str], layer_idx: int) -> List[int]:
        # 获取每个 uuid 请求的 offset，用于 mlx framework 旋转位置编码
        return [self.get_kv_len(uuid, layer_idx) for uuid in uuid_list]

    def update_cat(
        self,
        key_states: MIX_TENSOR,
        value_states: MIX_TENSOR,
        uuid_list: List[str],
        layer_idx: int,
        empty_k_cache: Optional[MIX_TENSOR] = None,
        empty_v_cache: Optional[MIX_TENSOR] = None,
    ):
        key_lst, value_lst = [], []
        start = 0
        for uuid in uuid_list:
            kv_cache: KVCache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            interval = self.get_q_len(uuid)
            end = start + interval
            cur_key_states, cur_value_states = key_states[start:end], value_states[start:end]

            if kv_cache.key_states is None:
                kv_cache.key_states, kv_cache.value_states = cur_key_states, cur_value_states
            else:
                kv_cache.key_states = cat_func([kv_cache.key_states, cur_key_states])
                kv_cache.value_states = cat_func([kv_cache.value_states, cur_value_states])

            kv_cache.set_kv_len(kv_cache.key_states.shape[0])
            key_lst.append(kv_cache.key_states)
            value_lst.append(kv_cache.value_states)
            start = end
        return cat_func(key_lst), cat_func(value_lst)

    def update_no_cat(
        self,
        key_states: MIX_TENSOR,
        value_states: MIX_TENSOR,
        uuid_list: List[str],
        layer_idx: int,
        empty_k_cache: MIX_TENSOR,
        empty_v_cache: MIX_TENSOR,
    ):
        assert empty_k_cache is not None and empty_v_cache is not None
        start = 0
        total_start = 0

        for uuid in uuid_list:
            kv_cache: KVCache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            end = start + self.get_q_len(uuid)  # 获取每个请求对应的区间
            cur_key_states, cur_value_states = key_states[start:end], value_states[start:end]

            if kv_cache.kv_len == 0:
                kv_cache.key_states[: end - start], kv_cache.value_states[: end - start] = (
                    cur_key_states,
                    cur_value_states,
                )
                kv_cache.set_kv_len(end - start)  # 更新 kv cache 的有效长度
            else:
                len_ = kv_cache.kv_len
                (
                    kv_cache.key_states[len_ : len_ + 1],
                    kv_cache.value_states[len_ : len_ + 1],
                ) = (cur_key_states, cur_value_states)
                kv_cache.set_kv_len(len_ + 1)  # 更新 kv cache 的有效长度

            start = end  # 更新下一个请求的起始位置

            # 最后拼接为整体
            total_end = total_start + kv_cache.kv_len
            empty_k_cache[total_start:total_end] = kv_cache.key_states[: kv_cache.kv_len]
            empty_v_cache[total_start:total_end] = kv_cache.value_states[: kv_cache.kv_len]

            total_start = total_end
        return empty_k_cache[:total_end], empty_v_cache[:total_end]

    def update_tinygrad(self, key_states, value_states, uuid_list, layer_idx):
        key_lst, value_lst = [], []
        start = 0

        for uuid in uuid_list:
            kv_cache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            interval = self.get_q_len(uuid)
            end = start + interval

            cur_key, cur_value = key_states[start:end], value_states[start:end]

            if kv_cache.key_states is None:
                kv_cache.key_states, kv_cache.value_states = cur_key, cur_value
            else:
                kv_cache.key_states = kv_cache.key_states.cat(cur_key, dim=0)
                kv_cache.value_states = kv_cache.value_states.cat(cur_value, dim=0)

            kv_cache.set_kv_len(kv_cache.key_states.shape[0])
            key_lst.append(kv_cache.key_states)
            value_lst.append(kv_cache.value_states)
            start = end

        return key_lst[0].cat(*key_lst[1:], dim=0), value_lst[0].cat(*value_lst[1:], dim=0)


@dataclass
class AttentionData:
    uuid_list: List[str]
    request_cache: RequestsCache
    attn_mask: MIX_TENSOR
