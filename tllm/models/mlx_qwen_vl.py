from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig, AutoProcessor

from tllm.commons.mlx_layers import PatchEmbed, PatchMerger, VisionMlp, VisionRotaryEmbedding, VisionSdpaAttention
from tllm.models.mlx_helper import quantization_func, read_state_dict, tie_embedding_weights
from tllm.models.utils import get_model_path, read_eos_token_ids


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: AutoConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionSdpaAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for thw in grid_thw:
            t, h, w = thw.tolist()
            hpos_ids = mx.repeat(mx.expand_dims(mx.arange(h), axis=1), w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.repeat(mx.expand_dims(mx.arange(w), axis=0), h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(mx.repeat(mx.stack([hpos_ids, wpos_ids], axis=-1), t, axis=1))
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        repeated = mx.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = mx.cumsum(repeated)
        cu_seqlens = mx.pad(cu_seqlens, pad_width=(1, 0)).tolist()

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


class MLXQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dtype = mx.bfloat16
        self.visual = Qwen2VisionModel(config.vision_config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, logger, config, model_path: str, state_dict: Optional[Any] = None):
        model = cls(config)

        # load processor
        model_path = get_model_path(model_path)
        model.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.mm_config = {
            "vision_start_id": config.vision_start_token_id,
            "vision_end_id": config.vision_end_token_id,
            "image_token_id": config.image_token_id,
            "video_token_id": config.video_token_id,
        }

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.logger = logger
        cls.eos_token_ids = read_eos_token_ids(config)

        if state_dict is None:
            state_dict = read_state_dict(model_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.layers"):
                continue
            if k == "visual.patch_embed.proj.weight":
                v = v.transpose(0, 2, 3, 4, 1)
            new_state_dict[k.split("model.")[-1]] = v
        state_dict = tie_embedding_weights(new_state_dict)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(
        self,
        x: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
        pixel_values_videos: Optional[np.ndarray] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        video_grid_thw: Optional[np.ndarray] = None,
    ) -> mx.array:
        input_ids = mx.array(x)
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = mx.array(pixel_values).astype(self.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=mx.array(image_grid_thw))
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            image_mask = input_ids == self.config.image_token_id  # shape: (seq_len, )
            image_mask_ind = [i for i, val in enumerate(image_mask) if val]
            image_embeds = image_embeds.astype(inputs_embeds.dtype)

            inputs_embeds[image_mask_ind] = image_embeds  # mlx not support bool mask

        if pixel_values_videos is not None:
            pixel_values_videos = mx.array(pixel_values_videos).astype(self.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=mx.array(video_grid_thw))
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_mask = input_ids == self.config.video_token_id  # shape: (seq_len, )
            video_mask_ind = [i for i, val in enumerate(video_mask) if val]
            video_embeds = video_embeds.astype(inputs_embeds.dtype)
            inputs_embeds[video_mask_ind] = video_embeds  # mlx not support bool mask

        return inputs_embeds

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states))
        return logits
