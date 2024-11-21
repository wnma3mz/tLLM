import itertools
import time
from typing import *

import torch

from tllm import HAS_MLX
from tllm.schemas import MIX_TENSOR

from tllm.commons.attn import get_attention_implementation


if HAS_MLX:
    import mlx.core as mx

    seq_dim = -2
    cat_func = lambda tensors: mx.concat(tensors, axis=seq_dim)
    split_func = lambda tensor, indices: mx.split(tensor, indices, axis=seq_dim)
else:
    _, attention_type, seq_dim = get_attention_implementation()
    cat_func = lambda tensors: torch.cat(tensors, dim=seq_dim)
    split_func = lambda tensor, indices: torch.split(tensor, indices, dim=seq_dim)


KV_CACHE_TYPE = Tuple[MIX_TENSOR, MIX_TENSOR]


class KVCache:
    def __init__(self) -> None:
        # key_states/value_states: num_heads x seq_len x head_dim OR seq_len x num_heads x head_dim
        self.key_states: Optional[MIX_TENSOR] = None
        self.value_states: Optional[MIX_TENSOR] = None

    def __len__(self) -> int:
        return 0 if self.key_states is None else self.key_states.shape[seq_dim]

    def update(self, key_states: MIX_TENSOR, value_states: MIX_TENSOR) -> KV_CACHE_TYPE:
        if self.key_states is not None:
            key_states = cat_func([self.key_states, key_states])
            value_states = cat_func([self.value_states, value_states])
        self.key_states, self.value_states = key_states, value_states
        return key_states, value_states


class RequestsCache:
    def __init__(self, num_layers: int) -> None:
        self.cache_dict: Dict[str : Dict[str, Union[List[KVCache], int]]] = {}
        self.num_layers = num_layers

    def get_seq_len_list(self, uuid_list: List[str]) -> List[int]:
        # 获取每个 uuid 请求的 seq_len，用于 split key_states/value_states
        return [self.get_seq_len(uuid) for uuid in uuid_list]

    def get_index_list(self, uuid_list: List[str]) -> List[int]:
        # 获取每个 uuid 请求的 seq_len，用于 split key_states/value_states。for MLX framework
        return list(itertools.accumulate(self.get_seq_len_list(uuid_list)[:-1]))

    def add(self, uuid: str, seq_len: int, layer_cache_list: Optional[List[KVCache]] = None):
        # 保存每个 uuid 请求所有层的 cache
        self.cache_dict[uuid] = {
            "cache": [KVCache() for _ in range(self.num_layers)] if layer_cache_list is None else layer_cache_list,
            "seq_len": seq_len,
        }

    def get_kv_cache(self, uuid: str) -> List[KVCache]:
        return self.cache_dict[uuid]["cache"]

    def get_layer_idx_kv_cache(self, uuid: str, layer_idx: int) -> KVCache:
        return self.get_kv_cache(uuid)[layer_idx]

    def get_seq_len(self, uuid: str) -> int:
        # 获取每个 uuid 请求的 key_states/value_states 的 seq_len
        return self.cache_dict[uuid]["seq_len"]

    def get_cache_seq_len(self, uuid: str, layer_idx: Optional[int] = 0) -> int:
        # 获取每个 uuid 请求的 kv cache 的 seq_len
        return len(self.get_kv_cache(uuid)[layer_idx])

    def get_offset_list(self, uuid_list: List[str], layer_idx: int) -> List[int]:
        # 获取每个 uuid 请求的 offset，用于 mlx framework 旋转位置编码
        return [self.get_cache_seq_len(uuid, layer_idx) for uuid in uuid_list]

    def update(
        self,
        key_states: Union[torch.Tensor, List["mx.array"]],
        value_states: MIX_TENSOR,
        uuid_list: List[str],
        layer_idx: int,
    ) -> KV_CACHE_TYPE:
        if HAS_MLX:
            seq_key_states = key_states  # 已经在外部 split 过了
            seq_value_states = split_func(value_states, self.get_index_list(uuid_list))
        else:
            seq_key_states = split_func(key_states, self.get_seq_len_list(uuid_list))
            seq_value_states = split_func(value_states, self.get_seq_len_list(uuid_list))

        key_states_list, value_states_list = [], []
        for uuid, key_state, value_state in zip(uuid_list, seq_key_states, seq_value_states):
            kv_cache: KVCache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            key, value = kv_cache.update(key_state, value_state)
            key_states_list.append(key)
            value_states_list.append(value)

        return cat_func(key_states_list), cat_func(value_states_list)


class AttentionData:
    def __init__(
        self,
        uuid_list: List[str],
        request_cache: RequestsCache,
        attn_mask: MIX_TENSOR,
        position_ids: Optional[torch.Tensor] = None,
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
