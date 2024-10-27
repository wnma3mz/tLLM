from dataclasses import dataclass, field
from typing import *

import torch
from transformers.cache_utils import DynamicCache

try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    HAS_MLX = True
except:
    HAS_MLX = False


@dataclass
class AttentionCache:
    uuid_str_list: List[str]
    past_key_value: Union[DynamicCache, List[Any]]
    attn_mask: Union[torch.Tensor, "mx.array"]
    position_ids: Optional[torch.Tensor] = field(default=None, repr=False)


class SeqDynamicCache:
    def __init__(self) -> None:
        self.cache_dict: Dict[Any] = {}

    def add(self, uuid_str: str, seq_len: int, cache: Optional[DynamicCache] = None):
        self.cache_dict.update({uuid_str: {"cache": DynamicCache() if cache is None else cache, "seq_len": seq_len}})

    def get_cache(self, uuid_str: str) -> DynamicCache:
        return self.cache_dict[uuid_str]["cache"]

    def get_seq_len(self, uuid_str: str) -> int:
        return self.cache_dict[uuid_str]["seq_len"]

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


def make_prompt_cache(num_layers: int, max_kv_size: Optional[int] = None) -> List[Any]:

    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]
    else:
        return [KVCache() for _ in range(num_layers)]


if HAS_MLX:

    class SeqMLXDynamicCache(SeqDynamicCache):
        def __init__(self, num_layers: int) -> None:
            super().__init__()
            self.num_layers = num_layers

        def add(self, uuid_str: str, seq_len: int, cache: Optional[DynamicCache] = None):
            self.cache_dict.update(
                {
                    uuid_str: {
                        "cache": make_prompt_cache(self.num_layers, max_kv_size=None) if cache is None else cache,
                        "seq_len": seq_len,
                    }
                }
            )

        def update_and_fetch(
            self,
            key_states: mx.array,
            value_states: mx.array,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[mx.array, mx.array]:
            # for mlx
            # TODO
            return key_states, value_states
