from functools import partial
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen2_5_vl import VisionConfig as Qwen2_5VisionConfig, VisionModel as Qwen2_5VisionModel
from mlx_vlm.models.qwen2_vl import VisionConfig as Qwen2VisionConfig, VisionModel as Qwen2VisionModel
from mlx_vlm.models.qwen3_vl import (
    Model as qwen3_vl_model,
    VisionConfig as Qwen3VisionConfig,
    VisionModel as Qwen3VisionModel,
)
import numpy as np
from transformers import AutoProcessor

from tllm import DTYPE
from tllm.models.mlx.helper import quantization_func
from tllm.models.utils import default_process_mm_input, merge_mm_input, read_from_text_config
from tllm.models.weight_helper import tie_word_embeddings_func


class MLXQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = getattr(config, "vocab_size", None) or read_from_text_config(config, "vocab_size")
        self.hidden_size = getattr(config, "hidden_size", None) or read_from_text_config(config, "hidden_size")
        self.rms_norm_eps = getattr(config, "rms_norm_eps", None) or read_from_text_config(config, "rms_norm_eps")

        if config.model_type == "qwen2_5_vl":
            self.vision_tower = Qwen2_5VisionModel(Qwen2_5VisionConfig.from_dict(config.vision_config.to_dict()))
            self.get_input_embeddings = self.get_input_embeddings_qwen2
        elif config.model_type == "qwen2_vl":
            self.vision_tower = Qwen2VisionModel(Qwen2VisionConfig.from_dict(config.vision_config.to_dict()))
            self.get_input_embeddings = self.get_input_embeddings_qwen2
        elif config.model_type == "qwen3_vl" or config.model_type == "qwen3_vl_moe":
            self.vision_tower = Qwen3VisionModel(Qwen3VisionConfig.from_dict(config.vision_config.to_dict()))
            self.get_input_embeddings = self.get_input_embeddings_qwen3

            # patch mlx_vlm func for qwen3_vl
            self.language_model = type("obj", (object,), {})
            self.language_model.model = type("obj", (object,), {})
            self.language_model.model.embed_tokens = None
            self.merge_input_ids_with_image_features = qwen3_vl_model.merge_input_ids_with_image_features
            self.qwen3_vl_get_input_embeddings = qwen3_vl_model.get_input_embeddings
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self.merge_mm_input = merge_mm_input

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, mx.array], **kwargs):
        assert kwargs.get("model_path", None) is not None

        model = cls(config)
        processor = AutoProcessor.from_pretrained(kwargs["model_path"], trust_remote_code=True)
        model.process_mm_input = partial(default_process_mm_input, image_processor=processor.image_processor)

        cls.config = config
        cls.config.image_token_index = config.image_token_id
        cls.config.video_token_index = config.video_token_id
        cls.num_layers = getattr(config, "num_hidden_layers", None) or read_from_text_config(
            config, "num_hidden_layers"
        )

        state_dict = tie_word_embeddings_func(config, state_dict)
        state_dict = model.sanitize(state_dict)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

        if hasattr(model, "language_model"):
            model.language_model.model.embed_tokens = model.embed_tokens
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
            # if k.startswith("vision_tower."):
            #     k = k.replace("vision_tower.", "visual.")

            if k == "vision_tower.patch_embed.proj.weight":
                # [out_ch, in_ch, n, h, w] -> [out_ch, n, h, w, in_ch]
                if v.shape[3] == v.shape[4]:
                    v = v.transpose(0, 2, 3, 4, 1)

            sanitized_weights[k] = v
        return sanitized_weights

    def get_input_embeddings_qwen2(
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
            image_embeds = self.vision_tower(pixel_values, grid_thw=mx.array(image_grid_thw))
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
            video_embeds = self.vision_tower(pixel_values_videos, grid_thw=mx.array(video_grid_thw))
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

    def get_input_embeddings_qwen3(
        self,
        input_ids: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
        pixel_values_videos: Optional[np.ndarray] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        video_grid_thw: Optional[np.ndarray] = None,
    ) -> mx.array:
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if pixel_values is not None:
            pixel_values = mx.array(pixel_values)
        if grid_thw is not None:
            grid_thw = mx.array(grid_thw)

        inputs_embeds, _, _ = self.qwen3_vl_get_input_embeddings(self, mx.array(input_ids), pixel_values, grid_thw)
        return inputs_embeds

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states))
        return logits
