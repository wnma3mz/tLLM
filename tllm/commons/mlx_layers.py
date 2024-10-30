from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs, TransformerBlock

from tllm.models.cache import AttentionCache, SeqMLXDynamicCache


class MyAttention(Attention):
    def _rope(self, xs: mx.array, seq_cache: SeqMLXDynamicCache) -> List[mx.array]:
        index_list = seq_cache.index_list
        offset_list = seq_cache.offset_list
        return [self.rope(x, offset) for x, offset in zip(mx.split(xs, index_list, axis=-2), offset_list)]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[AttentionCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # must has cache, and split by uuid_str
        seq_cache = cache.past_key_value
        queries = mx.concat(self._rope(queries, seq_cache), axis=-2)
        keys = self._rope(keys, seq_cache)

        cache_kwargs = {"uuid_str_list": cache.uuid_str_list}
        keys, values = seq_cache.update_and_fetch(keys, values, cache_kwargs)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MyTransformerBlock(TransformerBlock):
    def __init__(self, args: ModelArgs):
        super(TransformerBlock).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = MyAttention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args
