import base64
from functools import partial
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_clip.models.siglip.siglip_model import SiglipVisionModel
import numpy as np

from tllm import DTYPE
from tllm.models.mlx.helper import dict_to_dataclass, quantization_func
from tllm.models.mlx.vq_model import ModelArgs, VQModel, vision_head
from tllm.models.processor import VLMImageProcessor


def replace_vision_model_func(k: str, prefix_key: str) -> Optional[str]:
    k = k.split("vision_model.", 1)[-1]
    if f"{prefix_key}blocks." in k:
        k = k.replace(f"{prefix_key}blocks.", f"{prefix_key}encoder.layers.")
    if f"{prefix_key}patch_embed.proj." in k:
        k = k.replace(f"{prefix_key}patch_embed.proj.", f"{prefix_key}embeddings.patch_embedding.")

    # do not load attn_pool
    if "attn_pool." in k:
        k = k.replace("attn_pool.", "head.")

    if ".norm2." in k:
        k = k.replace(".norm2.", ".layer_norm2.")
    if ".norm1." in k:
        k = k.replace(".norm1.", ".layer_norm1.")
    if ".attn." in k:
        k = k.replace(".attn.", ".self_attn.")
    return k


def merge_mm_input(mm_input_list: List[Dict[str, List[np.ndarray]]]) -> Optional[Dict[str, List[mx.array]]]:
    if all([x is None for x in mm_input_list]) or all([len(x) == 0 for x in mm_input_list]):
        return None
    pixel_values_list = []
    for x in mm_input_list:
        if "image" in x:
            pixel_values_list.append(x["image"]["pixel_values"])

    pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else None

    return {
        "pixel_values": pixel_values,
    }


def process_mm_input(multi_modal_inputs: Dict[str, Union[List, str]], image_processor) -> Dict[str, List[np.ndarray]]:
    multi_modal_dict = {}
    multi_modal_dict["text"] = multi_modal_inputs["text"]

    if multi_modal_inputs is None and len(multi_modal_inputs) == 1:
        return multi_modal_dict

    images = multi_modal_inputs.get("image", None)
    repeat_times = 576  # fix value

    if images:
        if not isinstance(images, list):
            images = [images]
        new_imgs = []
        for img in images:
            new_imgs.append(img.convert("RGB"))
        images = new_imgs

        image_inputs = image_processor(images=images, videos=None)
        # 全部放到开头
        image_input_text = "<begin_of_image>" + "<image_placeholder>" * repeat_times + "<end_of_image>\n"
        multi_modal_dict["text"] = image_input_text * len(images) + multi_modal_dict["text"]
        multi_modal_dict.update({"image": image_inputs})

    return multi_modal_dict


class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        mlp_depth = cfg.get("depth", 1)
        layers = [nn.Linear(cfg["input_dim"], cfg["n_embed"])]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        """
        Inputs shape: []
        Output shape: [b, s, c]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MLXJanusProConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.aligner = MlpProjector(config.aligner_config["params"])

        language_config = dict_to_dataclass(config.language_config, "LanguageConfig")
        self.vocab_size = language_config.vocab_size
        self.embed_tokens = nn.Embedding(language_config.vocab_size, language_config.hidden_size)
        self.lm_head = nn.Linear(language_config.hidden_size, language_config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(language_config.hidden_size, eps=language_config.rms_norm_eps)

        self.gen_vision_model = VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4]))
        self.gen_aligner = MlpProjector(config.gen_aligner_config["params"])
        self.gen_head = vision_head(config.gen_head_config["params"])
        self.gen_embed = nn.Embedding(
            config.gen_vision_config["params"]["image_token_size"], config.gen_vision_config["params"]["n_embed"]
        )

        self.merge_mm_input = merge_mm_input
        self.img_size = 384
        self.patch_size = 16
        self.codebook_len = 8
        self.patch_img_size = self.img_size // self.patch_size

    @property
    def image_token_len(self):
        return self.patch_img_size * self.patch_img_size

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, mx.array], **kwargs):
        assert kwargs.get("model_path", None) is not None

        model = cls(config)
        image_processor = VLMImageProcessor.from_pretrained(kwargs["model_path"], trust_remote_code=True)
        model.process_mm_input = partial(process_mm_input, image_processor=image_processor)
        cls.image_token_id = config.image_token_id  # <image_placeholder>
        cls.pad_token_id = config.pad_token_id  #  <｜▁pad▁｜>
        cls.begin_image_token_id = config.begin_image_token_id  #  "<begin_of_image>"

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        state_dict = model.sanitize(state_dict)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        # Ugly compatibility janus
        for k, v in weights.items():
            if k.startswith("vision_model."):
                k = replace_vision_model_func(k, prefix_key="vision_tower.")
                # Skip attn_pool
                if k.startswith("vision_tower.head."):
                    continue
            if k.startswith("language_model."):
                k = k.replace("language_model.model.", "")
                k = k.replace("language_model.", "")

            if k.startswith("gen_vision_model."):
                if "weight" in k and len(v.shape) == 4:
                    # [out_ch, in_ch, h, w] -> [out_ch, h, w, in_ch]
                    v = v.transpose(0, 2, 3, 1)
                if "encoder" in k:
                    continue
                if ".quant_conv" in k:
                    continue

            if "codebook_used" in k:
                continue

            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # pytorch conv2d expects the weight tensor to be of shape [out_channels, in_channels, kH, KW]
                # mlx conv2d expects the weight tensor to be of shape [out_channels, kH, KW, in_channels]
                sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    def get_input_embeddings(
        self,
        input_ids: np.ndarray,
        pixel_values: Optional[np.ndarray] = None,
    ) -> mx.array:
        # TODO: Multi-Request Maybe Has Problem
        inputs_embeds = self.embed_tokens(mx.array(input_ids))

        if pixel_values is not None:
            # for mlx framework need to transpose
            # bs, c, h, w -> bs, h, w, c
            pixel_values = pixel_values.transpose(0, 2, 3, 1)

            pixel_values = mx.array(pixel_values).astype(DTYPE)
            image_embeds = self.aligner(self.vision_tower(pixel_values))
            # image_embeds: token_nums x hidden_size
            image_embeds = image_embeds[0]  # TODO: fix this

            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = input_ids == self.image_token_id  # shape: (seq_len, )
            image_mask_ind = [i for i, val in enumerate(image_mask) if val]
            image_embeds = image_embeds.astype(inputs_embeds.dtype)

            inputs_embeds[image_mask_ind] = image_embeds  # mlx not support bool mask
        return inputs_embeds

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states))
        return logits

    def get_gen_head(self, hidden_states: mx.array, temperature: float = 1.0, cfg_weight: float = 5.0) -> mx.array:
        logits = self.gen_head(self.norm(hidden_states))
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = mx.softmax(logits / temperature, axis=-1)
        return probs

    def get_gen_img_embeds(self, input_ids: np.ndarray) -> mx.array:
        return self.gen_aligner(self.gen_embed(mx.array(input_ids)))

    def decode_image(self, input_ids: mx.array) -> str:
        parallel_size = 1
        # only one image
        dec = self.gen_vision_model.decode_code(
            mx.array(input_ids),
            shape=[parallel_size, self.codebook_len, self.patch_img_size, self.patch_img_size],
            channel_first=True,
        )[0]
        dec = dec.astype(mx.float32)
        mlx_dec = mx.clip((dec + 1) / 2 * 255, 0, 255)
        np_dec = np.array(mlx_dec, dtype=np.uint8)
        return base64.b64encode(np_dec.tobytes()).decode("utf-8")
