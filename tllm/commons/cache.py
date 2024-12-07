# coding: utf-8
import time
from typing import Dict, List, Optional, Tuple, Union

from tllm import BACKEND, DTYPE, BackendEnum
from tllm.schemas import MIX_TENSOR

if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    cat_func = lambda tensors: mx.concat(tensors, axis=0)
    zeros_func = lambda x0, x1, x2: mx.zeros(shape=(x0, x1, x2), dtype=DTYPE)
else:
    import torch

    cat_func = lambda tensors: torch.cat(tensors, dim=0)
    zeros_func = lambda x0, x1, x2: torch.zeros(size=(x0, x1, x2), dtype=DTYPE)


KV_CACHE_TYPE = Tuple[MIX_TENSOR, MIX_TENSOR]


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
        self._len = 0

    def set_kv_len(self, len_: int):
        self._len = len_

    @property
    def kv_len(self):
        return self._len


class RequestsCache:
    def __init__(
        self, num_layers: int, max_seq_len: int = -1, num_key_value_heads: int = -1, head_dim: int = -1
    ) -> None:
        self.cache_dict: Dict[str : Dict[str, Union[List[KVCache], int]]] = {}
        self.num_layers = num_layers
        self.max_seq_len, self.num_key_value_heads, self.head_dim = max_seq_len, num_key_value_heads, head_dim
        if self.max_seq_len == -1:
            # not cat attention to save memory
            self.update = self.update_cat
        else:
            # cat attention to save time
            self.update = self.update_no_cat

    def add(self, uuid: str, seq_len: int, layer_cache_list: Optional[List[KVCache]] = None):
        # 保存每个 uuid 请求所有层的 cache
        self.cache_dict[uuid] = {
            "cache": (
                [KVCache(self.max_seq_len, self.num_key_value_heads, self.head_dim) for _ in range(self.num_layers)]
                if layer_cache_list is None
                else layer_cache_list
            ),
            "seq_len": seq_len,
        }

    def build(self, seq_input, cache_manager):
        q_len_list, k_len_list = [], []

        for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            if uuid in cache_manager.cache_dict:
                layer_cache_list, cache_seq_len = cache_manager.get(uuid)
                k_len_list.append(cache_seq_len + q_len)
            else:
                layer_cache_list = None
                k_len_list.append(q_len)
            q_len_list.append(q_len)

            self.add(uuid, q_len, layer_cache_list)
        return q_len_list, k_len_list

    def get_kv_cache(self, uuid: str) -> List[KVCache]:
        return self.cache_dict[uuid]["cache"]

    def get_layer_idx_kv_cache(self, uuid: str, layer_idx: int) -> KVCache:
        return self.get_kv_cache(uuid)[layer_idx]

    def get_seq_len(self, uuid: str) -> int:
        # 获取每个 uuid 请求的 key_states/value_states 的 seq_len
        return self.cache_dict[uuid]["seq_len"]

    def get_cache_seq_len(self, uuid: str, layer_idx: Optional[int] = 0) -> int:
        # 获取每个 uuid 请求的 kv cache 的 seq_len
        x = self.get_kv_cache(uuid)[layer_idx].kv_len
        return x

    def get_offset_list(self, uuid_list: List[str], layer_idx: int) -> List[int]:
        # 获取每个 uuid 请求的 offset，用于 mlx framework 旋转位置编码
        return [self.get_cache_seq_len(uuid, layer_idx) for uuid in uuid_list]

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
            interval = self.get_seq_len(uuid)
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
        empty_k_cache: Optional[MIX_TENSOR] = None,
        empty_v_cache: Optional[MIX_TENSOR] = None,
    ) -> KV_CACHE_TYPE:
        start = 0
        total_start = 0

        k_list, v_list = [], []
        for uuid in uuid_list:
            kv_cache: KVCache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            end = start + self.get_seq_len(uuid)  # 获取每个请求对应的区间
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
            if empty_k_cache is None:
                k_list.append(kv_cache.key_states[: kv_cache.kv_len])
                v_list.append(kv_cache.value_states[: kv_cache.kv_len])
            else:
                empty_k_cache[total_start:total_end] = kv_cache.key_states[: kv_cache.kv_len]
                empty_v_cache[total_start:total_end] = kv_cache.value_states[: kv_cache.kv_len]

            total_start = total_end
        if empty_k_cache is None:
            return cat_func(k_list), cat_func(v_list)
        else:
            return empty_k_cache[:total_end], empty_v_cache[:total_end]

    def update_tinygrad(self, key_states, value_states, uuid_list, layer_idx):
        key_lst, value_lst = [], []
        start = 0

        for uuid in uuid_list:
            kv_cache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            interval = self.get_seq_len(uuid)
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


class AttentionData:
    def __init__(
        self,
        uuid_list: List[str],
        request_cache: RequestsCache,
        attn_mask: MIX_TENSOR,
        position_ids=None,
    ) -> None:
        self.uuid_list = uuid_list
        self.request_cache = request_cache
        self.attn_mask = attn_mask
        self.position_ids = position_ids  # 只在 torch 下有意义

    def get_kv_cache_list(self, uuid: str) -> List[KVCache]:
        return self.request_cache.get_kv_cache(uuid)

    def get_cache_seq_len(self, uuid: str) -> int:
        return self.request_cache.get_cache_seq_len(uuid)


class CacheManager:
    # 管理每个节点的 cache kv_cache
    # max_alive_time: 超过多久没有访问就删除，单位秒
    def __init__(self, max_alive_time=60):
        self.max_alive_time = max_alive_time
        self.cache_dict = {}

    def get(self, key) -> Tuple[AttentionData, int]:
        return self.cache_dict.get(key)["cache"], self.cache_dict.get(key)["seq_len"]

    def set(self, key, value: List[KV_CACHE_TYPE], seq_len: int) -> None:
        self.cache_dict[key] = {"cache": value, "ts": time.time(), "seq_len": seq_len}

    def delete(self, key):
        self.cache_dict.pop(key)

    def clear(self):
        self.cache_dict.clear()

    def check_alive(self):
        now = time.time()
        key_list = list(self.cache_dict.keys())
        for key in key_list:
            if now - self.cache_dict[key]["ts"] > self.max_alive_time:
                self.cache_dict.pop(key)
