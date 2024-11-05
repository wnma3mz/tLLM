from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs, TransformerBlock

from tllm.models.cache import AttentionData, RequestsCache, cat_func, split_func


class MyAttention(Attention):
    def __init__(self, args, layer_idx: int, offset: int):
        super().__init__(args)
        self.layer_idx = layer_idx
        self.offset = offset

    def _rope(self, xs: mx.array, request_cache: RequestsCache, uuid_list: List[str]) -> List[mx.array]:
        index_list = request_cache.get_index_list(uuid_list)
        offset_list = request_cache.get_offset_list(uuid_list, self.layer_idx - self.offset)
        return [self.rope(x, offset) for x, offset in zip(split_func(xs, index_list), offset_list)]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: AttentionData,
    ) -> mx.array:
        L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(L, self.n_heads, -1).transpose(1, 0, 2)
        keys = keys.reshape(L, self.n_kv_heads, -1).transpose(1, 0, 2)
        values = values.reshape(L, self.n_kv_heads, -1).transpose(1, 0, 2)

        # must has cache, and split by uuid
        request_cache: RequestsCache = cache.request_cache
        queries = cat_func(self._rope(queries, request_cache, cache.uuid_list))
        keys = self._rope(keys, request_cache, cache.uuid_list)

        cache_kwargs = {"uuid_list": cache.uuid_list, "layer_idx": self.layer_idx - self.offset}
        keys, values = request_cache.update(keys, values, **cache_kwargs)

        output = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(queries, axis=0),
            mx.expand_dims(keys, axis=0),
            mx.expand_dims(values, axis=0),
            scale=self.scale,
            mask=mask,
        )[0]
        output = output.transpose(1, 0, 2).reshape(L, -1)

        return self.o_proj(output)


class MyTransformerBlock(TransformerBlock):
    def __init__(self, args: ModelArgs, layer_idx: int, offset: int):
        super(TransformerBlock).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = MyAttention(args, layer_idx, offset)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args
