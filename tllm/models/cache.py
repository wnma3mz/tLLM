from dataclasses import dataclass, field
import time
from typing import *

import torch
from transformers.cache_utils import DynamicCache

try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    HAS_MLX = True
except:
    HAS_MLX = False


@dataclass
class AttentionCache:
    uuid_str_list: List[str]
    past_key_value: Union[DynamicCache, "KVCache"]
    attn_mask: Union[torch.Tensor, "mx.array"]
    position_ids: Optional[torch.Tensor] = field(default=None, repr=False)


class SeqDynamicCache:
    def __init__(self) -> None:
        self.cache_dict: Dict[Any] = {}

    def add(self, uuid_str: str, seq_len: int, cache: Optional[DynamicCache] = None):
        self.cache_dict.update({uuid_str: {"cache": DynamicCache() if cache is None else cache, "seq_len": seq_len}})

    def get_cache(self, uuid_str: str) -> Union[DynamicCache, "KVCache"]:
        return self.cache_dict[uuid_str]["cache"]

    def get_seq_len(self, uuid_str: str) -> int:
        return self.cache_dict[uuid_str]["seq_len"]

    def get_max_seq_len(self) -> int:
        return max(self.get_seq_len(uuid_str) for uuid_str in self.cache_dict.keys())

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
        uuid_str_list = cache_kwargs.get("uuid_str_list", None)
        seq_len_list = [self.get_seq_len(uuid_str) for uuid_str in uuid_str_list]
        seq_key_states = torch.split(key_states, seq_len_list, dim=-2)
        seq_value_states = torch.split(value_states, seq_len_list, dim=-2)

        key_states_list, value_states_list = [], []
        for uuid_str, key_states, value_states in zip(uuid_str_list, seq_key_states, seq_value_states):
            key, value = self.get_cache(uuid_str).update(key_states, value_states, layer_idx)
            key_states_list.append(key)
            value_states_list.append(value)

        cat_key_states, cat_value_states = torch.cat(key_states_list, dim=-2), torch.cat(value_states_list, dim=-2)
        return cat_key_states, cat_value_states


if HAS_MLX:

    class SeqMLXDynamicCache(SeqDynamicCache):
        def get_max_offset(self) -> int:
            return max(self.get_cache(uuid_str).offset for uuid_str in self.cache_dict.keys())

        def add(self, uuid_str: str, seq_len: int, cache: Optional[KVCache] = None):
            cache = KVCache() if cache is None else cache
            offset = cache.offset
            self.cache_dict.update({uuid_str: {"cache": cache, "seq_len": seq_len, "offset": offset}})

        def update_and_fetch(
            self,
            key_states: mx.array,
            value_states: mx.array,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[mx.array, mx.array]:
            # TODO test multi requests
            uuid_str_list = cache_kwargs.get("uuid_str_list", None)
            seq_len_list = [self.get_seq_len(uuid_str) for uuid_str in uuid_str_list]
            seq_key_states = mx.split(key_states, seq_len_list, axis=-2)
            seq_value_states = mx.split(value_states, seq_len_list, axis=-2)

            key_states_list, value_states_list = [], []
            for uuid_str, key_states, value_states in zip(uuid_str_list, seq_key_states, seq_value_states):
                key_states, value_states = self.get_cache(uuid_str).update_and_fetch(key_states, value_states)
                key_states_list.append(key_states)
                value_states_list.append(value_states)

            cat_key_states, cat_value_states = mx.concat(key_states_list, axis=-2), mx.concat(
                value_states_list, axis=-2
            )
            return cat_key_states, cat_value_states


class CacheManager:
    # 管理每个 client 的 past_key_values，即 kv_cache
    # max_alive_time: 超过多久没有访问就删除，单位秒
    def __init__(self, max_alive_time=60):
        self.max_alive_time = max_alive_time
        self.cache_dict = {}

    def get(self, key) -> Any:
        return self.cache_dict.get(key)["past_key_values"]

    def set(self, key, value: Any) -> None:
        self.cache_dict[key] = {"past_key_values": value, "ts": time.time()}

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
