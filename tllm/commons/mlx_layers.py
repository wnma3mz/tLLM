import itertools
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs, TransformerBlock, initialize_rope

from tllm.models.cache import AttentionData, RequestsCache, cat_func, split_func


class BaseParallelLayer(nn.Module):
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        super().__init__()


class MergeParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size: int, dup_layer: int, world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        assert col_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.dup_layer = dup_layer
        self.layer = nn.Linear(row_size, col_size * self.dup_layer // self.world_size, bias=False)
        self.ind = [i * col_size // self.world_size for i in range(1, self.dup_layer)]

    def __call__(self, x: mx.array) -> List[mx.array]:
        node_output = self.layer(x)
        return mx.split(node_output, self.ind, axis=-1)


class QKVParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size_list: List[int], world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        for x in col_size_list:
            assert x % self.world_size == 0
        col_size = sum(col_size_list)
        assert col_size % self.world_size == 0

        self.row_size, self.col_size = row_size, col_size
        self.col_size_list = [x // self.world_size for x in col_size_list]
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=False)
        self.ind = list(itertools.accumulate(self.col_size_list[:-1]))

    def __call__(self, x: mx.array) -> List[mx.array]:
        node_output = self.layer(x)
        return mx.split(node_output, self.ind, axis=-1)


class RowParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size: int, world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        assert row_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.layer = nn.Linear(row_size // self.world_size, col_size, bias=False)
        self.ind = [i * row_size // self.world_size for i in range(1, self.world_size)]

    def load_weight(self, w: Optional[mx.array] = None):
        if self.world_size > 1:
            w_list = w.split(self.ind, axis=1)
            w = w_list[self.rank]
        state_dict = {"layer.weight": w}
        self.load_weights(list(state_dict.items()))

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer(x)


class MyAttention(nn.Module):
    def __init__(self, args, layer_idx: int, offset: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.qkv_proj = QKVParallelLayer(dim, [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim], 1, 0)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(args)

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
        queries, keys, values = self.qkv_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(L, self.n_heads, -1).transpose(1, 0, 2)
        keys = keys.reshape(L, self.n_kv_heads, -1).transpose(1, 0, 2)
        values = values.reshape(L, self.n_kv_heads, -1).transpose(1, 0, 2)

        # must has cache, and split by uuid
        request_cache: RequestsCache = cache.request_cache
        queries = cat_func(self._rope(queries, request_cache, cache.uuid_list))
        keys = self._rope(keys, request_cache, cache.uuid_list)

        keys, values = request_cache.update(keys, values, cache.uuid_list, self.layer_idx - self.offset)

        output = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(queries, axis=0),
            mx.expand_dims(keys, axis=0),
            mx.expand_dims(values, axis=0),
            scale=self.scale,
            mask=mask,
        )[0]
        output = output.transpose(1, 0, 2).reshape(L, -1)

        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.gate_up_proj = MergeParallelLayer(dim, hidden_dim, 2, 1, 0)

    def __call__(self, x) -> mx.array:
        gate_out, up_out = self.gate_up_proj(x)
        return self.down_proj(nn.silu(gate_out) * up_out)


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
