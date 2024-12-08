from dataclasses import dataclass
import os
from typing import Tuple

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.weights.model_saver import ModelSaver
from mlx import nn
import mlx.core as mx
from transformers import CLIPTokenizer, T5Tokenizer

from tllm.models.mlx.flux.transformer import Transformer

# from mflux.models.transformer.transformer import Transformer
# from mflux.tokenizer.tokenizer_handler import TokenizerHandler


@dataclass
class EmbeddingResult:
    latents: mx.array
    prompt_embeds: mx.array
    pooled_prompt_embeds: mx.array


class TokenizerHandler:
    def __init__(
        self,
        root_path: str,
        max_t5_length: int = 256,
    ):

        self.clip = CLIPTokenizer.from_pretrained(
            os.path.join(root_path, "tokenizer"),
            local_files_only=True,
            max_length=TokenizerCLIP.MAX_TOKEN_LENGTH,
            trust_remote_code=True,
        )
        self.t5 = T5Tokenizer.from_pretrained(
            os.path.join(root_path, "tokenizer_2"),
            local_files_only=True,
            max_length=max_t5_length,
            trust_remote_code=True,
        )


class Flux1:
    # get_embedding -> [get_encoder_hidden_states -> rpc request -> get_noise] * t -> get_images
    def __init__(
        self,
        model_config: ModelConfig,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model_config = model_config

        # Load and initialize the tokenizers from disk, huggingface cache, or download from huggingface
        tokenizers = TokenizerHandler(model_config.model_name, self.model_config.max_sequence_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        self.vae = VAE()
        self.transformer = Transformer(model_config)
        self.t5_text_encoder = T5Encoder()
        self.clip_text_encoder = CLIPEncoder()

    @classmethod
    def from_pretrained(cls, config, state_dict, **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        model._quantization_func(kwargs.get("quantization_level", None), state_dict)
        # model.load_weights(list(state_dict.items()))

        # mx.eval(model.parameters())
        # model.eval()
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
            nn.quantize(self.vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=self.bits)
            nn.quantize(self.t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            # fmt: on
            self._set_model_weights(weights)

    def get_embedding(self, seed: int, prompt: str, config: Config) -> Tuple[RuntimeConfig, EmbeddingResult]:
        config = RuntimeConfig(Config(**config.dict()), self.model_config)

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(seed, config, self.vae)
        # 2. Embed the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        return config, EmbeddingResult(latents, prompt_embeds, pooled_prompt_embeds)

    def get_encoder_hidden_states(self, t: int, config, embedding_result: EmbeddingResult) -> mx.array:
        # 3.t Predict the noise
        return self.transformer.predict(
            t=t,
            prompt_embeds=embedding_result.prompt_embeds,
            pooled_prompt_embeds=embedding_result.pooled_prompt_embeds,
            hidden_states=embedding_result.latents,
            config=config,
        )

    def get_noise(self, t: int, config, noise, latents) -> mx.array:
        # 4.t Take one denoise step
        return latents + noise * (config.sigmas[t + 1] - config.sigmas[t])

    def get_images(
        self, latents, config: RuntimeConfig, seed: int, prompt: str, generation_time: float
    ) -> GeneratedImage:
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=generation_time,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            init_image_path=config.init_image_path,
            init_image_strength=config.init_image_strength,
            config=config,
        )

    @staticmethod
    def from_alias(alias: str, quantize: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_alias(alias),
            quantize=quantize,
        )

    def _set_model_weights(self, weights):
        self.vae.update(weights.vae)
        self.transformer.update(weights.transformer)
        self.t5_text_encoder.update(weights.t5_encoder)
        self.clip_text_encoder.update(weights.clip_encoder)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)
