from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from tllm import DEVICE, DTYPE


class HFQwen2VisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):
    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


class HFQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.visual = HFQwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=DEVICE)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=DEVICE)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, torch.Tensor], **kwargs):
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

        model.load_state_dict(state_dict)
        model.to(DTYPE).to(DEVICE)
        model.eval()
        return model

    @torch.inference_mode()
    def get_input_embeddings(
        self,
        x: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
        pixel_values_videos: Optional[np.ndarray] = None,
        image_grid_thw: Optional[np.ndarray] = None,
        video_grid_thw: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        input_ids = torch.tensor(x, device=DEVICE)
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = torch.tensor(pixel_values, dtype=inputs_embeds.dtype, device=DEVICE)
            image_embeds = self.visual(pixel_values, grid_thw=torch.tensor(image_grid_thw, device=DEVICE))
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
            pixel_values_videos = torch.tensor(pixel_values_videos, dtype=inputs_embeds.dtype, device=DEVICE)
            video_embeds = self.visual(pixel_values_videos, grid_thw=torch.tensor(video_grid_thw, device=DEVICE))
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

    @torch.inference_mode()
    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(DTYPE).to(DEVICE)
        logits = self.lm_head(self.norm(hidden_states))
        return logits
