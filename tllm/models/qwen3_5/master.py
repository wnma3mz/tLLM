from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5 import (
    Model as qwen3_5_model,
    VisionConfig as Qwen3_5VisionConfig,
    VisionModel as Qwen3_5VisionModel,
)
from mlx_vlm.models.qwen3_5_moe import (
    Model as qwen3_5_moe_model,
    VisionConfig as Qwen3_5MoEVisionConfig,
    VisionModel as Qwen3_5MoEVisionModel,
)
import numpy as np
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

from tllm import DTYPE
from tllm.models.backend_mlx.helper import quantization_func
from tllm.models.mm_utils import default_process_mm_input, merge_mm_input
from tllm.models.qwen3_5.vlm_adapter import build_qwen35_language_model_shim
from tllm.models.text_utils import read_from_text_config
from tllm.models.weight_helper import tie_word_embeddings_func


class MLXQwen35ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = getattr(config, "vocab_size", None) or read_from_text_config(config, "vocab_size")
        self.hidden_size = getattr(config, "hidden_size", None) or read_from_text_config(config, "hidden_size")
        self.rms_norm_eps = getattr(config, "rms_norm_eps", None) or read_from_text_config(config, "rms_norm_eps")

        is_moe = config.model_type == "qwen3_5_moe"
        if is_moe:
            self.vision_tower = Qwen3_5MoEVisionModel(Qwen3_5MoEVisionConfig.from_dict(config.vision_config.to_dict()))
            qwen_model = qwen3_5_moe_model
        else:
            self.vision_tower = Qwen3_5VisionModel(Qwen3_5VisionConfig.from_dict(config.vision_config.to_dict()))
            qwen_model = qwen3_5_model

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.merge_mm_input = merge_mm_input

        self.language_model = build_qwen35_language_model_shim(config, is_moe, self.embed_tokens)
        self.qwen35_get_input_embeddings = qwen_model.get_input_embeddings
        self.merge_input_ids_with_image_features = qwen_model.merge_input_ids_with_image_features
        self.get_input_embeddings = self.get_input_embeddings_qwen3

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], **kwargs):
        assert kwargs.get("model_path", None) is not None
        model = cls(config)

        processor_holder = {"image_processor": None, "supports_video": False}

        def _process_mm_input_with_lazy_processor(multi_modal_inputs):
            has_image_input = bool(multi_modal_inputs.get("image"))
            has_video_input = bool(multi_modal_inputs.get("video"))
            has_vision_input = has_image_input or has_video_input
            if has_vision_input and processor_holder["image_processor"] is None:
                if has_video_input:
                    # Video processing for qwen-vl depends on torch/torchvision.
                    processor = AutoProcessor.from_pretrained(kwargs["model_path"], trust_remote_code=True)
                    processor_holder["image_processor"] = processor.image_processor
                    processor_holder["supports_video"] = True
                else:
                    # Image-only path keeps MLX-only env usable without torch deps.
                    processor_holder["image_processor"] = AutoImageProcessor.from_pretrained(
                        kwargs["model_path"], trust_remote_code=True
                    )
                    processor_holder["supports_video"] = False

            if has_video_input and not processor_holder["supports_video"]:
                raise RuntimeError(
                    "Video input requires torch/torchvision in current processor path. "
                    "Install them or send image-only requests."
                )
            return default_process_mm_input(multi_modal_inputs, image_processor=processor_holder["image_processor"])

        model.process_mm_input = _process_mm_input_with_lazy_processor

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

            if k == "vision_tower.patch_embed.proj.weight" and v.shape[3] == v.shape[4]:
                v = v.transpose(0, 2, 3, 4, 1)
            sanitized_weights[k] = v
        return sanitized_weights

    def get_input_embeddings_qwen3(
        self,
        input_ids: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
        pixel_values_videos: Optional[np.ndarray] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        video_grid_thw: Optional[np.ndarray] = None,
    ) -> mx.array:
        input_ids = mx.array(input_ids)
        squeeze_batch = False
        if input_ids.ndim == 1:
            # qwen3.5 rope index expects [batch, seq] input ids.
            input_ids = input_ids[None, :]
            squeeze_batch = True

        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if pixel_values is not None:
            pixel_values = mx.array(pixel_values).astype(DTYPE)
        if grid_thw is not None:
            grid_thw = mx.array(grid_thw)

        output = self.qwen35_get_input_embeddings(
            self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        )
        inputs_embeds = output.inputs_embeds if hasattr(output, "inputs_embeds") else output[0]
        if squeeze_batch and inputs_embeds.ndim == 3 and inputs_embeds.shape[0] == 1:
            inputs_embeds = inputs_embeds.squeeze(0)
        return inputs_embeds

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        return self.lm_head(self.norm(hidden_states))
