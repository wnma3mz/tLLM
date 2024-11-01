from dataclasses import dataclass, field
import time
from typing import *

import torch
from transformers.cache_utils import DynamicCache

try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    HAS_MLX = True
    CACHE_CLASS = KVCache
    cat_func = mx.concat
    split_func = mx.split
except:
    CACHE_CLASS = DynamicCache
    cat_func = torch.cat
    split_func = torch.split
    HAS_MLX = False


class SeqDynamicCache:
    def __init__(self) -> None:
        self.cache_dict: Dict[Any] = {}

    def add(self, uuid: str, seq_len: int, cache: Optional[DynamicCache] = None):
        self.cache_dict.update({uuid: {"cache": CACHE_CLASS() if cache is None else cache, "seq_len": seq_len}})

    def get_cache(self, uuid: str) -> Union[DynamicCache, "KVCache"]:
        return self.cache_dict[uuid]["cache"]

    def get_seq_len(self, uuid: str) -> int:
        return self.cache_dict[uuid]["seq_len"]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # key_states: bsz x num_heads x seq_len x head_dim
        uuid_list = cache_kwargs.get("uuid_list", None)
        seq_len_list = [self.get_seq_len(uuid) for uuid in uuid_list]
        seq_key_states = torch.split(key_states, seq_len_list, dim=-2)
        seq_value_states = torch.split(value_states, seq_len_list, dim=-2)

        key_states_list, value_states_list = [], []
        for uuid, key_state, value_state in zip(uuid_list, seq_key_states, seq_value_states):
            key, value = self.get_cache(uuid).update(key_state, value_state, layer_idx)
            key_states_list.append(key)
            value_states_list.append(value)

        cat_key_states, cat_value_states = cat_func(key_states_list, dim=-2), cat_func(value_states_list, dim=-2)
        return cat_key_states, cat_value_states


class SeqMLXDynamicCache(SeqDynamicCache):
    def add(self, uuid: str, seq_len: int, cache: Optional["KVCache"] = None):
        cache = CACHE_CLASS() if cache is None else cache
        offset = cache.offset
        self.cache_dict.update({uuid: {"cache": cache, "seq_len": seq_len, "offset": offset}})

    @property
    def offset_list(self) -> List[int]:
        return [self.get_cache(uuid).offset for uuid in self.cache_dict.keys()]

    @property
    def index_list(self) -> List[int]:
        seq_len_list = [self.get_seq_len(uuid) for uuid in self.cache_dict.keys()]
        index_list, idx = [], 0
        for seq_len in seq_len_list[:-1]:
            idx += seq_len
            index_list.append(idx)
        return index_list

    def update_and_fetch(
        self,
        seq_key_states: List["mx.array"],
        value_states: "mx.array",
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["mx.array", "mx.array"]:
        uuid_list = cache_kwargs.get("uuid_list", None)
        seq_value_states = split_func(value_states, self.index_list, axis=-2)
        key_states_list, value_states_list = [], []
        for uuid, key_state, value_state in zip(uuid_list, seq_key_states, seq_value_states):
            key, value = self.get_cache(uuid).update_and_fetch(key_state, value_state)
            key_states_list.append(key)
            value_states_list.append(value)

        cat_key_states, cat_value_states = cat_func(key_states_list, axis=-2), cat_func(value_states_list, axis=-2)
        return cat_key_states, cat_value_states


MIX_TENSOR = Union[torch.Tensor, "mx.array"]
KV_CACHE_TYPE = Tuple[MIX_TENSOR, MIX_TENSOR]


@dataclass
class AttentionCache:
    uuid_list: List[str]
    past_key_value: Union[SeqDynamicCache, SeqMLXDynamicCache]
    attn_mask: Union[torch.Tensor, "mx.array"]
    position_ids: Optional[torch.Tensor] = field(default=None, repr=False)


class NextKVCache:
    def __init__(self) -> None:
        # key_states/value_states: bsz x num_heads x seq_len x head_dim
        self.key_states: Optional[MIX_TENSOR] = None
        self.value_states: Optional[MIX_TENSOR] = None

    def __len__(self) -> int:
        return 0 if self.key_states is None else self.key_states.shape[-2]

    def update(self, key_states: MIX_TENSOR, value_states: MIX_TENSOR) -> KV_CACHE_TYPE:
        if self.key_states is not None:
            if HAS_MLX:
                key_states = cat_func([self.key_states, key_states], axis=-2)
                value_states = cat_func([self.value_states, value_states], axis=-2)
            else:
                key_states = cat_func([self.key_states, key_states], dim=-2)
                value_states = cat_func([self.value_states, value_states], dim=-2)
        self.key_states, self.value_states = key_states, value_states
        return key_states, value_states


class NextRequestsCache:
    def __init__(self, num_layers: int) -> None:
        self.cache_dict: Dict[str : Dict[str, Union[List[NextKVCache], int]]] = {}
        self.num_layers = num_layers

    def seq_len_list(self, uuid_list: List[str]) -> List[int]:
        return [self.get_seq_len(uuid) for uuid in uuid_list]

    def index_list(self, uuid_list: List[str]) -> List[int]:
        index_list, idx = [], 0
        for seq_len in self.seq_len_list(uuid_list)[:-1]:
            idx += seq_len
            index_list.append(idx)
        return index_list

    def add(self, uuid: str, seq_len: int, layer_cache_list: Optional[List[NextKVCache]] = None):
        # 保存每个 uuid 所有层的 cache
        self.cache_dict[uuid] = {
            "cache": [NextKVCache() for _ in range(self.num_layers)] if layer_cache_list is None else layer_cache_list,
            "seq_len": seq_len,
        }

    def get_kv_cache(self, uuid: str) -> List[NextKVCache]:
        return self.cache_dict[uuid]["cache"]

    def get_layer_idx_kv_cache(self, uuid: str, layer_idx: int) -> NextKVCache:
        return self.get_kv_cache(uuid)[layer_idx]

    def get_seq_len(self, uuid: str) -> int:
        return self.cache_dict[uuid]["seq_len"]

    def get_cache_seq_len(self, uuid: str) -> int:
        return len(self.get_kv_cache(uuid)[0])

    def update(
        self, key_states: Union[torch.Tensor, List["mx.array"]], value_states: MIX_TENSOR, **cache_kwargs
    ) -> KV_CACHE_TYPE:
        assert "uuid_list" in cache_kwargs
        assert "layer_idx" in cache_kwargs
        uuid_list = cache_kwargs.get("uuid_list", None)
        layer_idx = cache_kwargs.get("layer_idx", 0)
        if HAS_MLX:
            seq_key_states = key_states  # 已经在外部 split 过了
            seq_value_states = split_func(value_states, self.index_list(uuid_list), axis=-2)
        else:
            seq_key_states = split_func(key_states, self.seq_len_list(uuid_list), dim=-2)
            seq_value_states = split_func(value_states, self.seq_len_list(uuid_list), dim=-2)

        key_states_list, value_states_list = [], []
        for uuid, key_state, value_state in zip(uuid_list, seq_key_states, seq_value_states):
            kv_cache: NextKVCache = self.get_layer_idx_kv_cache(uuid, layer_idx)
            key, value = kv_cache.update(key_state, value_state)
            key_states_list.append(key)
            value_states_list.append(value)

        if HAS_MLX:
            return cat_func(key_states_list, axis=-2), cat_func(value_states_list, axis=-2)
        else:
            return cat_func(key_states_list, dim=-2), cat_func(value_states_list, dim=-2)


class NextAttentionData:
    def __init__(
        self,
        uuid_list: List[str],
        request_cache: NextRequestsCache,
        attn_mask: MIX_TENSOR,
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.uuid_list = uuid_list
        self.request_cache = request_cache  # 每层模型都有一个 NextDynamicCache
        self.attn_mask = attn_mask
        self.position_ids = position_ids  # 只在 torch 下有意义

    def get_kv_cache_list(self, uuid: str) -> List[NextKVCache]:
        return self.request_cache.get_kv_cache(uuid)

    def get_cache_seq_len(self, uuid: str) -> int:
        return self.request_cache.get_cache_seq_len(uuid)


class CacheManager:
    # 管理每个 client 的 cache kv_cache
    # max_alive_time: 超过多久没有访问就删除，单位秒
    def __init__(self, max_alive_time=60):
        self.max_alive_time = max_alive_time
        self.cache_dict = {}

    def get(self, key) -> Tuple[NextAttentionData, int]:
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
