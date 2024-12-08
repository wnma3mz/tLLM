import math
from typing import List

from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.transformer.embed_nd import EmbedND
from mflux.models.transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.transformer.single_transformer_block import SingleTransformerBlock
from mflux.models.transformer.time_text_embed import TimeTextEmbed
from mlx import nn
import mlx.core as mx

from tllm.commons.cache import CacheManager


class SingleTransformer(nn.Module):
    def __init__(self, start_idx: int = 0, end_idx: int = 38, num_hidden_layers: int = 38):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_hidden_layers = num_hidden_layers

        self.pos_embed = EmbedND()
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(start_idx, end_idx)]
        if self.num_hidden_layers == self.end_idx:
            self.norm_out = AdaLayerNormContinuous(3072, 3072)
            self.proj_out = nn.Linear(3072, 64)

    def __call__(
        self, hidden_states, text_embeddings, image_rotary_emb, seq_len: int, controlnet_single_block_samples=None
    ):
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = block.forward(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            # if controlnet_single_block_samples is not None and len(controlnet_single_block_samples) > 0:
            #     interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            #     interval_control = int(math.ceil(interval_control))
            #     hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
            #         hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            #         + controlnet_single_block_samples[idx // interval_control]
            #     )
        if self.num_hidden_layers == self.end_idx:
            hidden_states = hidden_states[:, seq_len:, ...]
            return self.get_noise(hidden_states, text_embeddings)
        return hidden_states

    def get_noise(self, hidden_states, text_embeddings) -> mx.array:
        hidden_states = self.norm_out.forward(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class FLUXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_hidden_layers = 38
        self.transformer = SingleTransformer(0, self.num_hidden_layers, self.num_hidden_layers)
        self.embedding_cache_dict = CacheManager()

    @classmethod
    def from_pretrained(cls, config, state_dict, **kwargs):
        model = cls()

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        model._quantization_func(kwargs.get("quantization_level", None), state_dict)

        mx.eval(model.parameters())
        model.eval()
        return model

    def _quantization_func(self, quantization_level, weights):
        # Set the loaded weights if they are not quantized
        if quantization_level is None:
            self._set_model_weights(weights)

        # Optionally quantize the model here at initialization (also required if about to load quantized weights)
        self.bits = None
        if quantization_level is not None:
            self.bits = quantization_level
            # fmt: off
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=self.bits)
            # fmt: on
            self._set_model_weights(weights)

    def __call__(
        self,
        hidden_states,
        text_embeddings,
        seq_len: int,
        height: int,
        width: int,
        request_id_list: List[str],
        controlnet_single_block_samples=None,
    ) -> mx.array:

        # TODO: 对于每个请求，不需要重新计算
        txt_ids = Transformer.prepare_text_ids(seq_len=seq_len)
        img_ids = Transformer.prepare_latent_image_ids(height, width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.transformer.pos_embed.forward(ids)

        hidden_states = self.transformer(hidden_states, text_embeddings, image_rotary_emb, seq_len)
        mx.eval(hidden_states)
        return hidden_states

    def _set_model_weights(self, weights):
        self.transformer.update(weights.transformer)

    @staticmethod
    def prepare_latent_image_ids(height: int, width: int) -> mx.array:
        latent_width = width // 16
        latent_height = height // 16
        latent_image_ids = mx.zeros((latent_height, latent_width, 3))
        latent_image_ids = latent_image_ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
        latent_image_ids = latent_image_ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
        latent_image_ids = mx.repeat(latent_image_ids[None, :], 1, axis=0)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_width * latent_height, 3))
        return latent_image_ids

    @staticmethod
    def prepare_text_ids(seq_len: mx.array) -> mx.array:
        return mx.zeros((1, seq_len, 3))


class Transformer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(64, 3072)
        self.time_text_embed = TimeTextEmbed(model_config=model_config)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(19)]

    def predict(
        self,
        t: int,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        hidden_states: mx.array,
        config: RuntimeConfig,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
    ) -> mx.array:
        time_step = config.sigmas[t] * config.num_train_steps
        time_step = mx.broadcast_to(time_step, (1,)).astype(config.precision)
        hidden_states = self.x_embedder(hidden_states)
        guidance = mx.broadcast_to(config.guidance * config.num_train_steps, (1,)).astype(config.precision)
        text_embeddings = self.time_text_embed.forward(time_step, pooled_prompt_embeds, guidance)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        txt_ids = Transformer.prepare_text_ids(seq_len=prompt_embeds.shape[1])
        img_ids = Transformer.prepare_latent_image_ids(config.height, config.width)
        ids = mx.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed.forward(ids)

        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=image_rotary_emb,
            )
            if controlnet_block_samples is not None and len(controlnet_block_samples) > 0:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(math.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[idx // interval_control]

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        return hidden_states, text_embeddings

    @staticmethod
    def prepare_latent_image_ids(height: int, width: int) -> mx.array:
        latent_width = width // 16
        latent_height = height // 16
        latent_image_ids = mx.zeros((latent_height, latent_width, 3))
        latent_image_ids = latent_image_ids.at[:, :, 1].add(mx.arange(0, latent_height)[:, None])
        latent_image_ids = latent_image_ids.at[:, :, 2].add(mx.arange(0, latent_width)[None, :])
        latent_image_ids = mx.repeat(latent_image_ids[None, :], 1, axis=0)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_width * latent_height, 3))
        return latent_image_ids

    @staticmethod
    def prepare_text_ids(seq_len: mx.array) -> mx.array:
        return mx.zeros((1, seq_len, 3))
