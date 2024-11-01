import math
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm.commons.mlx_layers import MyTransformerBlock
from tllm.models.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.protocol import SeqInput


def build_mlx_mask(seq_len_list: List[Tuple[int, int]]) -> mx.array:
    mask_list = [
        mx.tril(mx.ones((L, S), dtype=mx.bool_), k=0) if L > 1 else mx.ones((L, S), dtype=mx.bool_)
        for (L, S) in seq_len_list
    ]

    total_L = sum(L for L, S in seq_len_list)
    total_S = sum(S for L, S in seq_len_list)
    combined_mask = mx.zeros((total_L, total_S), dtype=mx.bool_)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.shape[0], r_index : r_index + mask.shape[1]] = mask
        l_index += mask.shape[0]
        r_index += mask.shape[1]

    final_mask = mx.where(combined_mask, 0, -math.inf)
    return final_mask


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    actual_seq_len_list = []
    for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            actual_seq_len_list.append([seq_len, cache_seq_len + 1])
        else:
            layer_cache_list = None
            actual_seq_len_list.append([seq_len, seq_len])
        request_cache.add(uuid, seq_len, layer_cache_list)
    return AttentionData(
        request_cache=request_cache,
        attn_mask=build_mlx_mask(actual_seq_len_list),
        uuid_list=seq_input.uuid_list,
    )


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, start_layer_idx: int, end_layer_idx: int):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = [
            MyTransformerBlock(args=args, layer_idx=layer_idx, offset=start_layer_idx)
            for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, h: mx.array, mask, cache: AttentionData):
        for layer in self.layers:
            h = layer(h, mask, cache=cache)
        return h


class MyMLXLlamaModel(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        args = ModelArgs.from_dict(config.to_dict())
        self.vocab_size = args.vocab_size
        self.cache_manager = CacheManager()
        self.args = args
        self.model = Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> np.ndarray:
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_layers)

        mask = attention_data.attn_mask
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = self.model(hidden_states, mask=mask, cache=attention_data)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()
        return np.array(output.astype(mx.float16))

    @property
    def dtype(self):
        return next(self.parameters()).dtype
