import glob
import os
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm.commons.cache import AttentionData, CacheManager
from tllm.commons.mlx_layers import MLXTransformerBlock
from tllm.models.mlx_helper import (
    build_forward_cache,
    default_merge_attn_weight,
    default_merge_mlp_weight,
    empty_func,
    get_last_hidden_states,
    pop_weight_func,
    quantization_func,
    read_main_state_dict,
    read_state_dict,
)
from tllm.models.utils import get_model_path, read_eos_token_ids
from tllm.schemas import SeqInput


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = [empty_func] * start_layer_idx + [
            MLXTransformerBlock(args=args, layer_idx=layer_idx, offset=start_layer_idx, is_merge=is_merge)
            for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, h: mx.array, mask, cache: AttentionData) -> mx.array:
        for layer in self.layers:
            h = layer(h, mask, cache=cache)
        mx.eval(h)  # just for debug test
        return h


class MLXLlamaModel(nn.Module):
    def __init__(self, config: AutoConfig, is_merge: bool = True):
        super().__init__()
        args = ModelArgs.from_dict(config.to_dict())
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
    def from_pretrained(
        cls, config: AutoConfig, model_path: str, state_dict: Optional[Any] = None, is_merge: bool = True
    ):
        if getattr(config, "quantization", None) is not None or state_dict is not None:
            is_merge = False
        model = cls(config, is_merge)
        if state_dict is None:
            model_path = get_model_path(model_path)
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
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
        weights = pop_weight_func(
            prefix_key_list,
            weights,
            self.config.num_hidden_layers,
            self.config.decoder_start_layer_idx,
            self.config.decoder_end_layer_idx,
        )
        if not is_merge:
            return weights

        weights = default_merge_attn_weight(weights)
        weights = default_merge_mlp_weight(weights)

        return weights


class MLXLlamaForCausalLM(nn.Module):
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
        cls.eos_token_ids = read_eos_token_ids(config)

        if state_dict is None:
            state_dict = read_state_dict(model_path)

        state_dict = read_main_state_dict(state_dict)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states.astype(self.dtype)))
        return logits
