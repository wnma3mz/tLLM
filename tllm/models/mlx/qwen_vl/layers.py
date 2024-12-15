import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from tllm import DTYPE


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self._freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))

    def __call__(self, seqlen: int) -> mx.array:
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
        self.proj = nn.Conv3d(
            in_channels=in_channels, out_channels=embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # [out_ch, in_ch, n, h, w] -> [out_ch, n, h, w, in_ch]
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = self.proj(hidden_states).reshape(-1, self.embed_dim)
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
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


def QuickGELUActivation(input: mx.array) -> mx.array:
    return input * mx.sigmoid(1.702 * input)


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        # self.act = nn.SiLU()
        self.act = QuickGELUActivation  # for qwenvl
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
    # tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = mx.expand_dims(mx.tile(mx.expand_dims(cos, axis=1), (1, 1, 2)), axis=0)
    sin = mx.expand_dims(mx.tile(mx.expand_dims(sin, axis=1), (1, 1, 2)), axis=0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.astype(orig_dtype)
    return output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, hidden_states: mx.array, cu_seqlens: List[int], rotary_pos_emb: mx.array = None) -> mx.array:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, axis=0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, axis=0), rotary_pos_emb)[0]

        attention_mask = mx.zeros(shape=(1, seq_length, seq_length), dtype=mx.bool_)
        for i in range(1, len(cu_seqlens)):
            l, r = cu_seqlens[i - 1], cu_seqlens[i]
            attention_mask[..., l:r, l:r] = True
        attention_mask = mx.where(attention_mask, 0, -math.inf).astype(DTYPE)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        attn_output = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(q, axis=0),
            mx.expand_dims(k, axis=0),
            mx.expand_dims(v, axis=0),
            scale=1 / math.sqrt(q.shape[-1]),
            mask=attention_mask,
        )[0]
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output
