from functools import partial
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen2_5_vl.vision import VisionConfig as Qwen2_5VisionConfig, VisionModel as Qwen2_5VisionModel
from mlx_vlm.models.qwen2_vl.vision import VisionConfig as Qwen2VisionConfig, VisionModel as Qwen2VisionModel
import numpy as np
from transformers import AutoProcessor

from tllm import DTYPE
from tllm.models.mlx.helper import quantization_func
from tllm.models.utils import default_process_mm_input, merge_mm_input
from tllm.models.weight_helper import tie_word_embeddings_func


def build_config(config, model_type: str):
    if model_type == "qwen2_5_vl":
        return Qwen2_5VisionConfig(
            depth=config.depth,
            out_hidden_size=config.out_hidden_size,
            num_heads=config.num_heads,
            in_channels=config.in_channels,  # in_channels
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            spatial_patch_size=config.spatial_patch_size,
            temporal_patch_size=config.temporal_patch_size,
            window_size=config.window_size,
            intermediate_size=getattr(config, "intermediate_size", None),
        )
    elif model_type == "qwen2_vl":
        return Qwen2VisionConfig(
            depth=config.depth,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            in_channels=config.in_channels,  # in_channels
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            spatial_patch_size=config.spatial_patch_size,
            temporal_patch_size=config.temporal_patch_size,
            mlp_ratio=config.mlp_ratio,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class MLXQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        vision_config = build_config(config.vision_config, config.model_type)
        if config.model_type == "qwen2_5_vl":
            self.visual = Qwen2_5VisionModel(vision_config)
        elif config.model_type == "qwen2_vl":
            self.visual = Qwen2VisionModel(vision_config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.merge_mm_input = merge_mm_input

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, mx.array], **kwargs):
        assert kwargs.get("model_path", None) is not None

        model = cls(config)
        processor = AutoProcessor.from_pretrained(kwargs["model_path"], trust_remote_code=True)
        model.process_mm_input = partial(default_process_mm_input, image_processor=processor.image_processor)

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        state_dict = tie_word_embeddings_func(config, state_dict)
        state_dict = model.sanitize(state_dict)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model.model.layers"):
                continue
            if k.startswith("model.layers"):
                continue
            if k.startswith("language_model.model."):
                k = k.replace("language_model.model.", "")
            if k.startswith("language_model."):
                k = k.replace("language_model.", "")

            if k.startswith("model."):
                k = k.replace("model.", "")
            if k.startswith("vision_tower."):
                k = k.replace("vision_tower.", "visual.")

            if k == "visual.patch_embed.proj.weight":
                # [out_ch, in_ch, n, h, w] -> [out_ch, n, h, w, in_ch]
                if v.shape[3] == v.shape[4]:
                    v = v.transpose(0, 2, 3, 4, 1)

            sanitized_weights[k] = v
        return sanitized_weights

    def get_input_embeddings(
        self,
        input_ids: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
        pixel_values_videos: Optional[np.ndarray] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        video_grid_thw: Optional[np.ndarray] = None,
    ) -> mx.array:
        # TODO: Multi-Request Maybe Has Problem
        inputs_embeds = self.embed_tokens(mx.array(input_ids))

        if pixel_values is not None:
            pixel_values = mx.array(pixel_values).astype(DTYPE)
            image_embeds = self.visual(pixel_values, grid_thw=mx.array(image_grid_thw))
            # image_embeds: token_nums x hidden_size
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
            pixel_values_videos = mx.array(pixel_values_videos).astype(DTYPE)
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
