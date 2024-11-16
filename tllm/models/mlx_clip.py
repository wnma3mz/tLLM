# coding: utf-8
# Modified by https://github.com/ml-explore/mlx-examples/blob/main/clip/model.py

from dataclasses import dataclass

# only vision model
from typing import *

import mlx.core as mx
import mlx.nn as nn

from tllm.commons.mlx_layers import VisionEmbeddings
from tllm.models.mlx_llama import Decoder


@dataclass
class CLIPVisionConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    layer_norm_eps: float


class ClipVisionModel(nn.Module):
    """Implements the vision encoder transformer from CLIP."""

    def __init__(
        self,
        config: CLIPVisionConfig,
        first_num_layers: int = -1,
    ):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = Decoder(config, 0, config.num_hidden_layers, False)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)

        for l in self.encoder.layers:
            x = l(x, mask=None)

        # Extract <CLS> token embedding
        pooler_output = self.post_layernorm(x[:, 0, :])
        return pooler_output
