from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm import DTYPE
from tllm.commons.cache import CacheManager, RequestsCache
from tllm.models.mlx.helper import (
    build_forward_cache,
    get_last_hidden_states,
    quantization_func,
    truncate_hidden_states,
)
from tllm.models.mlx.layers import Decoder
from tllm.models.weight_helper import default_merge_attn, default_merge_mlp, tie_word_embeddings_func
from tllm.schemas import SeqInput


class MLXQwen2Model(nn.Module):
    def __init__(self, config: AutoConfig, is_merge: bool = True):
        super().__init__()
        config_dict = config.to_dict()
        config_dict.pop("rope_scaling")  # TODO: remove this line
        comm = config.comm
        del config.comm
        args = ModelArgs.from_dict(config_dict)

        args.comm = comm
        self.world_size = args.comm.world_size
        self.rank = args.comm.rank

        args.attention_bias = True  # for qwen
        args.o_proj_bias = False  # for qwen
        self.vocab_size = args.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

        self.max_seq_len = getattr(self.model.layers[-1].self_attn, "max_seq_len", -1)
        self.n_kv_heads = self.model.layers[-1].self_attn.n_kv_heads
        self.head_dim = self.model.layers[-1].self_attn.head_dim
        self.request_cache = RequestsCache(self.num_layers, self.max_seq_len, self.n_kv_heads, self.head_dim)

        self.is_start_pp = self.config.decoder_start_layer_idx == 0
        self.is_end_pp = self.config.decoder_end_layer_idx == self.config.num_hidden_layers

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.request_cache)
        hit_cache_flag = any(x != -1 for x in attention_data.hit_cache_len_list)
        hidden_states = truncate_hidden_states(hit_cache_flag, self.is_start_pp, attention_data, hidden_states)

        mask = attention_data.attn_mask
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = self.model(hidden_states, mask=mask, cache=attention_data)

        for uuid in seq_input.uuid_list:
            self.cache_manager.set(uuid, attention_data.get_decoder_cache(uuid))
            self.cache_manager.check_alive()
        self.request_cache.clear()
        self.request_cache.insert_cache(seq_input)

        output = get_last_hidden_states(hit_cache_flag, self.is_end_pp, attention_data, output)
        return output

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], **kwargs):
        is_merge = True

        model = cls(config, is_merge)
        state_dict = model.merge_weights(state_dict, is_merge)

        state_dict = model.sanitize(state_dict, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights, start_idx: int, end_idx: int):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                k = k.replace("language_model.", "")
            if not k.startswith("model."):
                continue
            if "embed_tokens" in k or "model.norm" in k:
                continue
            if int(k.split("model.layers.", 1)[-1].split(".")[0]) not in range(start_idx, end_idx):
                continue

            sanitized_weights[k] = v
        return sanitized_weights

    def merge_weights(self, state_dict: Dict[str, mx.array], is_merge: bool = True) -> Dict[str, mx.array]:
        if not is_merge:
            return state_dict
        layer_name_mapper = {
            "self_attn.o_proj": "self_attn.o_proj.layer",
            "mlp.down_proj": "mlp.down_proj.layer",
        }
        key_list = list(state_dict.keys())
        for key in key_list:
            for s_key, t_key in layer_name_mapper.items():
                if s_key in key:
                    state_dict[key.replace(s_key, t_key)] = state_dict.pop(key)

        state_dict = default_merge_attn(state_dict)
        state_dict = default_merge_mlp(state_dict)
        return state_dict


class MLXQwen2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Optional[Dict], **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        state_dict = tie_word_embeddings_func(config, state_dict)
        state_dict = model.sanitize(state_dict)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("model.layers."):
                continue
            sanitized_weights[k.split("model.")[-1]] = v
        return sanitized_weights

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states.astype(DTYPE)))
        return logits
