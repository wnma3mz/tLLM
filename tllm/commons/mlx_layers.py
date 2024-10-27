from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs, TransformerBlock


class MyAttention(Attention):
    def __init__(self, args: ModelArgs, layer_idx: int, offset: int) -> None:
        super().__init__(args)
        self.layer_idx = layer_idx
        self.offset = offset

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)

            cache_kwargs = {"uuid_str_list": cache.uuid_str_list}
            keys, values = cache.past_key_value.update_and_fetch(
                keys, values, self.layer_idx - self.offset, cache_kwargs
            )
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MyTransformerBlock(TransformerBlock):
    def __init__(self, args: ModelArgs, layer_idx, offset):
        super(TransformerBlock).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = MyAttention(args, layer_idx, offset)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args
