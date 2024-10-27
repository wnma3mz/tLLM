from typing import *

import mlx.core as mx
import mlx.nn as nn

from tllm.commons.mlx_layers import MyTransformerBlock
from tllm.models.cache import AttentionCache, CacheManager, SeqMLXDynamicCache
from tllm.models.protocol import SeqInput
from tllm.models.utils import build_mask


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionCache:
    past_key_values = SeqMLXDynamicCache(num_layers)
    actual_seq_len_list = []
    for uuid_str, seq_len in zip(seq_input.uuid_str_list, seq_input.seq_len_list):
        if uuid_str in cache_manager.cache_dict:
            kv_cache = cache_manager.get(uuid_str)
            actual_seq_len_list.append([seq_len, kv_cache.get_seq_length() + 1])
        else:
            kv_cache = None
            actual_seq_len_list.append([seq_len, seq_len])
        past_key_values.add(uuid_str, seq_len, kv_cache)

    return AttentionCache(
        past_key_value=past_key_values,
        uuid_str_list=seq_input.uuid_str_list,
        attn_mask=build_mask(actual_seq_len_list),  # TODO mlx create_attention_mask
    )


class Decoder(nn.Module):
    def __init__(self, args, start_layer_idx: int, end_layer_idx: int):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = [
            MyTransformerBlock(args=args, layer_idx=layer_idx, offset=start_layer_idx)
            for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, h: mx.array, mask, cache):
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return h


class MyMLXLlamaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.cache_manager = CacheManager()
        self.args = args
        self.decoder = Decoder(args, args.decoder_start_layer_idx, args.decoder_end_layer_idx)
        self.num_layers = args.decoder_end_layer_idx - args.decoder_start_layer_idx

    # def load_state_dict(self, state_dict: Dict) -> None:
    #     self.decoder.load_state_dict(state_dict)

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        attention_cache = build_forward_cache(seq_input, self.cache_manager, self.num_layers)
        hidden_states = hidden_states.to(self.device)
        output = self.decoder(hidden_states, mask=attention_cache.attn_mask, cache=attention_cache)

        for uuid_str, seq_len in zip(seq_input.uuid_str_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid_str, output.past_key_values.get_cache(uuid_str))
            self.cache_manager.check_alive()
        return output.last_hidden_state

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
