
import glob
import itertools
import os
import re
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm.generate.token_utils import TokenizerUtils
from tllm.models.mlx_llama import quantization_func
from tllm.models.utils import load_master_weight
from tllm.commons.mlx_layers import VisionMlp, VisionSdpaAttention, PatchEmbed, PatchMerger, VisionRotaryEmbedding 


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionSdpaAttention(
            config.embed_dim, num_heads=config.num_heads
        )
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array:
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
        for t, h, w in grid_thw:
            hpos_ids = mx.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(mx.stack([hpos_ids, wpos_ids], axis=-1).repeat(t, 1))
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = mx.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=mx.int32
        )
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), value=0)

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
    def from_pretrained(cls, logger, config, tok: TokenizerUtils, model_path: str, state_dict: Optional[Any] = None):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.logger = logger
        cls.eos_token_ids = set()

        if hasattr(config, "eos_token_ids"):
            if isinstance(config.eos_token_id, list):
                cls.eos_token_ids |= set(config.eos_token_ids)
            else:
                cls.eos_token_ids.add(config.eos_token_id)

        if tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(tok.tokenizer.eos_token_id)
        eos_token = tok.tokenizer.decode(list(cls.eos_token_ids))
        cls.logger.debug(f"eos_token_ids: {cls.eos_token_ids}; Tokens: {eos_token}")

        if state_dict is None:
            state_dict = {}
            weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))
            for wf in weight_files:
                state_dict.update(mx.load(wf))

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.layers"):
                continue
            if k == "visual.patch_embed.proj.weight":
                v = v.transpose(0, 2, 3, 4, 1)
            new_state_dict[k.split("model.")[-1]] = v
        state_dict = new_state_dict
        has_key_list = list(state_dict.keys())
        # print(has_key_list)
        if "lm_head.weight" not in state_dict:
            for key in has_key_list:
                if key.startswith("embed_tokens."):
                    state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray, pixel_values: Optional[mx.array] = None, pixel_values_videos: Optional[mx.array] = None, image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,) -> mx.array:
        input_ids = mx.array(x)
        inputs_embeds = self.embed_tokens(input_ids)
    
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds
    

    def get_logits(self, hidden_states: mx.array, seq_len_list: List[int]) -> List[mx.array]:
        # 只取最后一个 token 的 hidden_states
        index_list = list(itertools.accumulate(seq_len_list[:-1]))
        seq_hidden_states = mx.split(hidden_states, index_list, axis=0)
        hidden_states = mx.concat([x[-1:, :] for x in seq_hidden_states], axis=0).astype(self.dtype)
        logits = self.lm_head(self.norm(hidden_states))
        ind = list(itertools.accumulate([1] * (len(seq_len_list) - 1)))
        return mx.split(logits, ind, axis=0)
    