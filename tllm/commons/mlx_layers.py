import itertools
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import MLP, Attention, ModelArgs, TransformerBlock, initialize_rope

from tllm.models.cache import AttentionData, RequestsCache, cat_func, split_func


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
        return mx.split(node_output, self.ind, axis=-1)


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

    def load_weight(self, w: Optional[mx.array] = None):
        if self.world_size > 1:
            w_list = w.split(self.ind, axis=1)
            w = w_list[self.rank]
        state_dict = {"layer.weight": w}
        self.load_weights(list(state_dict.items()))

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer(x)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self._freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))

    def forward(self, seqlen: int) -> mx.array:
        seq = mx.arange(seqlen, dtype=self._freqs.dtype)
        freqs = mx.outer(seq, self._freqs)
        return freqs


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        # Patchify using conv:
        # [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
        patch_embeddings = self.patch_embedding(x)
        # [batch_size, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        # Prepend <CLS> embeddings
        # [batch_size, 1, embed_dim]
        cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
        # [batch_size, num_patches + 1, embed_dim]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        # Add positional encoding
        embeddings += self.position_embedding.weight
        return embeddings


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None) -> mx.array:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = mx.zeros([1, seq_length, seq_length], device=q.device, dtype=mx.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = mx.fast.scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class MergedAttention(nn.Module):
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
        if hasattr(args, "o_proj_bias"):
            o_proj_bias = args.o_proj_bias
        else:
            o_proj_bias = False

        self.qkv_proj = QKVParallelLayer(
            dim, [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim], 1, 0, bias=attention_bias
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=o_proj_bias)

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


class PlainAttention(Attention):
    def __init__(self, args, layer_idx: int, offset: int):
        super().__init__(args)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.o_proj_bias)
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


class MergedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.gate_up_proj = MergeParallelLayer(dim, hidden_dim, 2, 1, 0, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        gate_out, up_out = self.gate_up_proj(x)
        return self.down_proj(nn.silu(gate_out) * up_out)


class TransformerBlock(TransformerBlock):
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
