from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_clip.models.qwen2vision.qwen2vision_model import Qwen2VisionModel
import numpy as np
from transformers import AutoProcessor

from tllm import DTYPE
from tllm.models.mlx.helper import quantization_func


class MLXQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.visual = Qwen2VisionModel(config.vision_config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, mx.array], **kwargs):
        assert kwargs.get("model_path", None) is not None

        model = cls(config)
        model.processor = AutoProcessor.from_pretrained(kwargs["model_path"], trust_remote_code=True)
        model.mm_config = {
            "vision_start_id": config.vision_start_token_id,
            "vision_end_id": config.vision_end_token_id,
            "image_token_id": config.image_token_id,
            "video_token_id": config.video_token_id,
        }

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

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
