# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
import glob
import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

# from mlx.core import linalg as LA
from transformers import SiglipVisionConfig


@dataclass
class SiglipVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    use_head: bool = True
    mlp_ratio: int = 4
    ignore_head: bool = True

    @classmethod
    def from_dict(cls, data_dict: Dict):
        return cls(
            num_hidden_layers=data_dict.get("num_hidden_layers", 12),
            hidden_size=data_dict.get("hidden_size", 768),
            intermediate_size=data_dict.get("hidden_size", 768) * data_dict.get("mlp_ratio", 4),
            num_attention_heads=data_dict.get("num_attention_heads", 12),
            num_channels=data_dict.get("num_channels", 3),
            image_size=data_dict.get("image_size", 224),
            patch_size=data_dict.get("patch_size", 16),
            layer_norm_eps=data_dict.get("layer_norm_eps", 1e-6),
        )


# Modify from CLIP
class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.qkv = nn.Linear(dims, dims * 3, bias=bias)
        self.proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, hidden_states, mask=None):
        B, L, D = hidden_states.shape

        queries, keys, values = self.qkv(hidden_states).reshape(B, L, 3, -1).transpose(2, 0, 1, 3)

        _, S, _ = keys.shape
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, self.num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)

        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.proj(values_hat)


# Copied from CLIP
class MLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


# Copied from CLIP
class EncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Add biases to the attention projections
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


# Copied from CLIP
class Encoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        target_dtype = self.patch_embedding.weight.dtype
        # shape = [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.astype(target_dtype))
        # patch_embeds (1, 14, 14, 1024)
        # [batch_size, h, w, embed_dim]
        embeddings = mx.flatten(patch_embeds, start_axis=1, end_axis=2)
        # embeddings = patch_embeds.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return embeddings


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.kv = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.latent_len = 1
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.latent = mx.zeros((1, self.latent_len, config.hidden_size))
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.pos_embed = None

        self.pool = "token"

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape

        # if self.pos_embed is not None:
        #     # FIXME interpolate
        #     x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        # TODO: maybe slow
        q_latent = mx.repeat(self.latent, B, axis=0)
        q = self.q(q_latent)
        q = q.reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        k, v = kv

        # q, k = self.q_norm(q), self.k_norm(k)

        # self attn
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(0, 2, 1, 3).reshape(B, self.latent_len, C)
        x = self.proj(x)
        # x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)  # patch_embed
        # self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder = Encoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.use_head = getattr(config, "use_head", True)

        self.no_embed_class = False
        self.num_prefix_tokens = 0

        grid_size = config.image_size // config.patch_size
        self.pos_embed = mx.zeros((1, grid_size * grid_size, config.hidden_size))

        self.head = None
        self.ignore_head = config.ignore_head
        # attn_pool
        if not self.ignore_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def _pos_embed(self, x: mx.array) -> mx.array:
        return x + self.pos_embed

    def __call__(self, x: mx.array):
        x = self.embeddings(x)
        x = self._pos_embed(x)

        mask = None
        x = self.encoder(x, mask)
        x = self.norm(x)

        return x if self.ignore_head else self.head(x)

    @staticmethod
    def _load_default_config(config_data):
        config_data.update(
            {
                "image_size": 384,
                "patch_size": 16,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "mlp_ratio": 4,
                "global_pool": "map",
                "use_checkpoint": False,
            }
        )
        return config_data

    @classmethod
    def from_pretrained(cls, path: str):
        import os

        with open(os.path.join(path, "config.json"), "r") as f:
            config_data = json.load(f)["vision_config"]["params"]
        config_data = cls._load_default_config(config_data)
        config = SiglipVisionConfig.from_dict(config_data)

        model = cls(config)
        weight_files = glob.glob(str(Path(path) / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        # Ugly compatibility janus
        for k, v in weights.items():
            k = k.split("vision_tower.", 1)[-1]
            if "blocks." in k:
                k = k.replace("blocks.", "encoder.layers.")
            if "patch_embed.proj." in k:
                k = k.replace("patch_embed.proj.", "embeddings.patch_embedding.")
            # if "norm." in k:
            #     k = k.replace("norm.", "post_layernorm.")
            if "attn_pool." in k:
                k = k.replace("attn_pool.", "head.")

            if ".norm2." in k:
                k = k.replace(".norm2.", ".layer_norm2.")
            if ".norm1." in k:
                k = k.replace(".norm1.", ".layer_norm1.")
            if ".attn." in k:
                k = k.replace(".attn.", ".self_attn.")

            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # pytorch conv2d expects the weight tensor to be of shape [out_channels, in_channels, kH, KW]
                # mlx conv2d expects the weight tensor to be of shape [out_channels, kH, KW, in_channels]
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights


if __name__ == "__main__":
    model = SiglipVisionModel.from_pretrained("siglip_model")

    shape = (1, 384, 384, 3)
    pixel_value = mx.random.normal(shape=shape)
    # import torch
    # pixel_value_torch = torch.load("x.pth")
    # pixel_value = mx.array(pixel_value_torch).transpose(0, 2, 3, 1)

    output = model(pixel_value)
    # print("output", output, output.shape)
