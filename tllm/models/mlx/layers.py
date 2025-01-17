import itertools
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs, TransformerBlock, initialize_rope

from tllm.commons.cache import AttentionData, RequestsCache, cat_func


class BaseParallelLayer(nn.Module):
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        super().__init__()


class MergeParallelLayer(BaseParallelLayer):
    def __init__(
        self, row_size: int, col_size: int, dup_layer: int, world_size: int, rank: int, bias: bool = False
    ) -> None:
        super().__init__(world_size, rank)
        assert col_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.dup_layer = dup_layer
        self.layer = nn.Linear(row_size, col_size * self.dup_layer // self.world_size, bias=bias)
        self.ind = [i * col_size // self.world_size for i in range(1, self.dup_layer)]

    def __call__(self, x: mx.array) -> List[mx.array]:
        node_output = self.layer(x)
        return (node_output[:, : self.ind[0]], node_output[:, self.ind[0] :])


class QKVParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size_list: List[int], world_size: int, rank: int, bias: bool = False) -> None:
        super().__init__(world_size, rank)
        for x in col_size_list:
            assert x % self.world_size == 0
        col_size = sum(col_size_list)
        assert col_size % self.world_size == 0

        self.row_size, self.col_size = row_size, col_size
        self.col_size_list = [x // self.world_size for x in col_size_list]
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=bias)
        self.ind = list(itertools.accumulate(self.col_size_list[:-1]))

    def __call__(self, x: mx.array) -> List[mx.array]:
        node_output = self.layer(x)
        return mx.split(node_output, self.ind, axis=-1)


class RowParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size: int, world_size: int, rank: int, bias: bool = False) -> None:
        super().__init__(world_size, rank)
        assert row_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.layer = nn.Linear(row_size // self.world_size, col_size, bias=bias)
        self.ind = [i * row_size // self.world_size for i in range(1, self.world_size)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@mx.compile
def sdap(q, k, v, scale, mask):
    """
    处理3维输入的scaled dot-product attention

    Args:
        q: [L, H, D] - sequence length, heads, head dimension
        k: [S, H_kv, D] - S is source length, H_kv is number of key/value heads
        v: Same shape as k
        scale: Scaling factor
        mask: Attention mask
    """
    q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    q = mx.multiply(scale, q)
    n_q_heads = q.shape[0]  # 查询头数
    n_kv_heads = k.shape[0]  # key/value头数

    if n_q_heads > n_kv_heads:  # grouped query attention
        n_repeats = n_q_heads // n_kv_heads
        L, _, D = q.shape
        q = mx.reshape(q, (n_kv_heads, n_repeats, q.shape[1], q.shape[2]))

        # k和v保持[S, H_kv, D]格式，在计算时添加一个维度
        scores = mx.matmul(q, mx.swapaxes(k[:, None], -1, -2))
        scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        out = mx.matmul(scores, v[:, None])
        # 展平结果为[L, H, D]
        out = mx.flatten(out, 0, 1)

    else:  # 标准注意力计算
        scores = mx.matmul(q, mx.swapaxes(k, -1, -2))
        scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        out = mx.matmul(scores, v)
    return out.transpose(1, 0, 2)


class MergedAttention(nn.Module):
    def __init__(self, args, layer_idx: int, offset: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        attention_bias = getattr(args, "attention_bias", False)
        o_proj_bias = getattr(args, "o_proj_bias", False)

        self.comm = args.comm
        self.rank = self.comm.rank
        self.world_size = self.comm.world_size

        self.qkv_proj = QKVParallelLayer(
            dim,
            [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim],
            self.world_size,
            self.rank,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLayer(n_heads * head_dim, dim, self.world_size, self.rank, bias=o_proj_bias)

        self.layer_idx = layer_idx
        self.offset = offset

        # max_seq_len = 1024
        # self._k_cache = mx.zeros(shape=(max_seq_len, args.num_key_value_heads, self.head_dim), dtype=self.o_proj.weight.dtype)
        # self._v_cache = mx.zeros(shape=(max_seq_len, args.num_key_value_heads, self.head_dim), dtype=self.o_proj.weight.dtype)
        self._k_cache, self._v_cache = None, None
        self.rope = initialize_rope(args)

    def _rope(self, xs: mx.array, request_cache: RequestsCache, uuid_list: List[str]) -> List[mx.array]:
        offset_list = request_cache.get_offset_list(uuid_list, self.layer_idx - self.offset)
        x_list = []
        start = 0
        for uuid, offset in zip(uuid_list, offset_list):
            end = start + request_cache.get_seq_len(uuid)
            x_list.append(self.rope(xs[start:end].transpose(1, 0, 2), offset).transpose(1, 0, 2))
            start = end
        return cat_func(x_list)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: AttentionData,
    ) -> mx.array:
        L, _ = x.shape
        queries, keys, values = self.qkv_proj(x)
        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(L, -1, self.head_dim)
        keys = keys.reshape(L, -1, self.head_dim)
        values = values.reshape(L, -1, self.head_dim)

        # must has cache, and split by uuid
        request_cache: RequestsCache = cache.request_cache

        # queries = apply_rotary_pos_emb_v2(queries, cache.cos, cache.sin)
        # keys = apply_rotary_pos_emb_v2(keys, cache.cos, cache.sin)
        queries = self._rope(queries, request_cache, cache.uuid_list)
        keys = self._rope(keys, request_cache, cache.uuid_list)
        keys, values = request_cache.update(
            keys, values, cache.uuid_list, self.layer_idx - self.offset, self._k_cache, self._v_cache
        )

        output = sdap(queries, keys, values, scale=self.scale, mask=mask)
        output = output.reshape(L, -1)

        attn_output = self.o_proj(output)
        return self.comm.all_reduce(attn_output)


class PlainAttention(Attention):
    def __init__(self, args, layer_idx: int, offset: int):
        super().__init__(args)
        o_proj_bias = getattr(args, "o_proj_bias", False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=o_proj_bias)
        self.layer_idx = layer_idx
        self.offset = offset

    def _rope(self, xs: mx.array, request_cache: RequestsCache, uuid_list: List[str]) -> List[mx.array]:
        offset_list = request_cache.get_offset_list(uuid_list, self.layer_idx - self.offset)
        x_list = []
        start = 0
        for uuid, offset in zip(uuid_list, offset_list):
            end = start + request_cache.get_seq_len(uuid)
            x_list.append(self.rope(xs[start:end].transpose(1, 0, 2), offset).transpose(1, 0, 2))
            start = end
        return cat_func(x_list)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        cache: AttentionData,
    ) -> mx.array:
        L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(L, self.n_heads, -1)
        keys = keys.reshape(L, self.n_kv_heads, -1)
        values = values.reshape(L, self.n_kv_heads, -1)

        # must has cache, and split by uuid
        request_cache: RequestsCache = cache.request_cache
        queries = self._rope(queries, request_cache, cache.uuid_list)
        keys = self._rope(keys, request_cache, cache.uuid_list)
        keys, values = request_cache.update(keys, values, cache.uuid_list, self.layer_idx - self.offset)

        output = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(queries.transpose(1, 0, 2), axis=0),
            mx.expand_dims(keys.transpose(1, 0, 2), axis=0),
            mx.expand_dims(values.transpose(1, 0, 2), axis=0),
            scale=self.scale,
            mask=mask,
        )[0]
        output = output.transpose(1, 0, 2).reshape(L, -1)

        return self.o_proj(output)


class MergedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        mlp_bias = getattr(args, "mlp_bias", False)

        self.comm = args.comm
        self.rank = self.comm.rank
        self.world_size = self.comm.world_size

        self.down_proj = RowParallelLayer(hidden_dim, dim, self.world_size, self.rank, bias=mlp_bias)
        self.gate_up_proj = MergeParallelLayer(dim, hidden_dim, 2, self.world_size, self.rank, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        gate_out, up_out = self.gate_up_proj(x)
        out = self.down_proj(nn.silu(gate_out) * up_out)
        return self.comm.all_reduce(out)


class MLXTransformerBlock(TransformerBlock):
    def __init__(self, args: ModelArgs, layer_idx: int, offset: int, is_merge: bool = True):
        super(TransformerBlock).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        if is_merge:
            self.self_attn = MergedAttention(args, layer_idx, offset)
            self.mlp = MergedMLP(args)
        else:
            self.self_attn = PlainAttention(args, layer_idx, offset)
            self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args


def empty_func(h, mask, cache):
    # TODO
    return h


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = [empty_func] * start_layer_idx + [
            MLXTransformerBlock(args=args, layer_idx=layer_idx, offset=start_layer_idx, is_merge=is_merge)
            for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, h: mx.array, mask, cache: AttentionData) -> mx.array:
        for layer in self.layers:
            h = layer(h, mask, cache=cache)
        return h
