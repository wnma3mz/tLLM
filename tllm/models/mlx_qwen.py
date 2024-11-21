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

from tllm.commons.cache import CacheManager
from tllm.models.mlx_helper import get_last_hidden_states, read_from_safetensors
from tllm.models.mlx_llama import Decoder, build_forward_cache, quantization_func
from tllm.models.utils import get_weight_path
from tllm.schemas import SeqInput


class MLXQwen2Model(nn.Module):
    def __init__(self, config: AutoConfig, is_merge: bool = True):
        super().__init__()
        config_dict = config.to_dict()
        config_dict.pop("rope_scaling")  # TODO: remove this line
        args = ModelArgs.from_dict(config_dict)
        args.attention_bias = True  # for qwen
        args.o_proj_bias = False  # for qwen
        self.vocab_size = args.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_layers)

        mask = attention_data.attn_mask
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = self.model(hidden_states, mask=mask, cache=attention_data)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            output = get_last_hidden_states(output, seq_input.seq_len_list)
        return output

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, config: AutoConfig, model_path: str, state_dict: Optional[Any] = None):
        is_merge = True
        if getattr(config, "quantization", None) is not None or state_dict is not None:
            is_merge = False
        model = cls(config, is_merge)
        if state_dict is None:
            weights = model.read_weight_from_model_path(model_path, is_merge)
        else:
            weights = state_dict

        model = quantization_func(config, model, weights)
        model.load_weights(list(weights.items()))  # strict=False
        if getattr(config, "quantization", None) is None:
            model.set_dtype(mx.bfloat16)

        mx.eval(model.parameters())
        model.eval()
        return model

    def read_weight_from_model_path(self, model_path: str, is_merge: bool = True) -> Dict[str, mx.array]:
        print(f"start_idx: {self.config.decoder_start_layer_idx}, end_idx: {self.config.decoder_end_layer_idx}")
        attn_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn")
        mlp_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.mlp")
        qkv_proj_list = ["q_proj", "k_proj", "v_proj"]
        gate_up_list = ["gate_proj", "up_proj"]
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head.", "visual."]

        prefix_key_list += [
            f"model.layers.{i}."
            for i in range(self.config.num_hidden_layers)
            if not (self.config.decoder_start_layer_idx <= i < self.config.decoder_end_layer_idx)
        ]
        key_list = list(weights.keys())
        for key in key_list:
            for prefix_key in prefix_key_list:
                if key.startswith(prefix_key):
                    weights.pop(key)
        if not is_merge:
            return weights

        key_list = list(weights.keys())

        attn_proj_w = {}  # layer_idx -> {qkv: weight}
        attn_proj_b = {}  # layer_idx -> {qkv: bias}
        mlp_w = {}
        for key in key_list:
            attn_res = attn_layer_idx_pattern.findall(key)
            mlp_res = mlp_layer_idx_pattern.findall(key)
            if attn_res:
                layer_idx = int(attn_res[0])
                if ".weight" in key:
                    if layer_idx not in attn_proj_w:
                        attn_proj_w[layer_idx] = {}
                elif ".bias" in key:
                    if layer_idx not in attn_proj_b:
                        attn_proj_b[layer_idx] = {}
            elif mlp_res:
                layer_idx = int(mlp_res[0])
                if layer_idx not in mlp_w:
                    mlp_w[layer_idx] = {}
            else:
                continue

            for qkv in qkv_proj_list:
                if qkv in key and ".weight" in key:
                    attn_proj_w[layer_idx].update({qkv: weights.pop(key)})
                elif qkv in key and ".bias" in key:
                    attn_proj_b[layer_idx].update({qkv: weights.pop(key)})
            for mlp in gate_up_list:
                if mlp in key:
                    mlp_w[layer_idx].update({mlp: weights.pop(key)})

            layer_weights = attn_proj_w.get(layer_idx, [])
            if len(layer_weights) == 3:
                name = f"model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
                weights[name] = mx.concatenate([layer_weights[qkv] for qkv in qkv_proj_list], axis=0)
                attn_proj_w.pop(layer_idx)

            layer_weights = attn_proj_b.get(layer_idx, [])
            if len(layer_weights) == 3:
                name = f"model.layers.{layer_idx}.self_attn.qkv_proj.layer.bias"
                weights[name] = mx.concatenate([layer_weights[qkv] for qkv in qkv_proj_list], axis=0)
                attn_proj_b.pop(layer_idx)

            layer_weights = mlp_w.get(layer_idx, [])
            if len(layer_weights) == 2:
                name = f"model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
                weights[name] = mx.concatenate([layer_weights[mlp] for mlp in gate_up_list], axis=0)
                mlp_w.pop(layer_idx)

        return weights


class MLXQwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dtype = mx.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, logger, config, model_path: str, state_dict: Optional[Any] = None):
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

        if state_dict is None:
            file_set, prefix_key_list = get_weight_path(model_path)
            state_dict = {}
            for file in file_set:
                weight_path = os.path.join(model_path, file)
                state_dict.update(read_from_safetensors(weight_path, prefix_key_list))

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.") and not k.startswith("model.layers."):
                new_state_dict[k.split("model.")[-1]] = v
        state_dict = new_state_dict
        has_key_list = list(state_dict.keys())
        if "lm_head.weight" not in state_dict:
            for key in has_key_list:
                if key.startswith("embed_tokens."):
                    state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        # 只取最后一个 token 的 hidden_states
        logits = self.lm_head(self.norm(hidden_states))
        return logits
