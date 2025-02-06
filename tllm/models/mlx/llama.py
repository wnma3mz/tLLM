from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm import DTYPE
from tllm.models.mlx.helper import MLXCacheManager, quantization_func
from tllm.models.mlx.layers import Decoder
from tllm.models.weight_helper import (
    default_merge_attn,
    default_merge_mlp,
    tensor_parallel_state_dict,
    tie_word_embeddings_func,
)
from tllm.schemas import SeqInput

cache_manager = MLXCacheManager()


def get_inv_freq_mx(dim, base):
    return 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.int32).astype(mx.float32) / dim))


class DynamicNTKScalingRoPE:
    def __init__(self, dims, max_position_embeddings, base, scale, rope_type, rope_scaling):
        self._freqs = get_inv_freq_mx(dims, base)
        self._freqs = mx.expand_dims(self._freqs, (0, 2))[0]

    def __call__(self, position_ids):
        position_mx_ids_expanded = position_ids[None, :]
        freqs = (self._freqs @ position_mx_ids_expanded).transpose(1, 0)

        emb_mx = mx.concatenate((freqs, freqs), axis=-1)
        return emb_mx.cos(), emb_mx.sin()


class MLXLlamaModel(nn.Module):
    def __init__(self, config: AutoConfig, is_merge: bool = True):
        super().__init__()
        comm = config.comm
        del config.comm

        if config.architectures[0] in ["JanusProConditionalGeneration"]:
            config_dict = config.language_config
        else:
            config_dict = config.to_dict()

        args = ModelArgs.from_dict(config_dict)

        args.comm = comm
        self.world_size = args.comm.world_size
        self.rank = args.comm.rank

        self.vocab_size = args.vocab_size
        self.config = config
        self.model = Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        # TODO: Custom RoPE
        # self.rotary_emb = DynamicNTKScalingRoPE(
        #     dims=args.head_dim or args.hidden_size // args.num_attention_heads,
        #     max_position_embeddings=args.max_position_embeddings,
        #     base=args.rope_theta,
        #     scale=1.0,
        #     rope_type="default",
        #     rope_scaling=1.0,
        # )

        max_seq_len = getattr(self.model.layers[-1].self_attn, "max_seq_len", -1)
        n_kv_heads = self.model.layers[-1].self_attn.n_kv_heads
        head_dim = self.model.layers[-1].self_attn.head_dim
        cache_manager.init_request_cache(self.num_layers, max_seq_len, n_kv_heads, head_dim)

        is_start_pp = self.config.decoder_start_layer_idx == 0
        is_end_pp = self.config.decoder_end_layer_idx == self.config.num_hidden_layers
        cache_manager.post_init(is_start_pp, is_end_pp)

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        hidden_states = cache_manager.build_forward_cache(hidden_states, seq_input)
        # cos, sin = self.rotary_emb(attention_data.position_ids)
        # attention_data.cos, attention_data.sin = mx.expand_dims(cos, axis=1), mx.expand_dims(sin, axis=1)
        output = self.model(hidden_states, cache=cache_manager.attn_data)

        # TODO 异步更新 cache
        cache_manager.update_cache(seq_input)

        output = cache_manager.get_last_hidden_states(output)
        return output

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], **kwargs):
        is_merge = True

        model = cls(config, is_merge)
        state_dict = model.sanitize(state_dict, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        state_dict = model.merge_weights(state_dict, is_merge)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights, start_idx: int, end_idx: int):
        sanitized_weights = {}
        for k, v in weights.items():
            if not k.startswith("model"):
                continue
            if "embed_tokens" in k or "model.norm" in k:
                continue
            if int(k.split("model.layers.", 1)[-1].split(".")[0]) not in range(start_idx, end_idx):
                continue
            sanitized_weights[k.split("language_model.", 1)[-1]] = v

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

        # tensor parallel
        state_dict = tensor_parallel_state_dict(state_dict, self.world_size, self.rank)
        state_dict = default_merge_attn(state_dict)
        state_dict = default_merge_mlp(state_dict)
        return state_dict


class MLXLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict, **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        state_dict = tie_word_embeddings_func(config, state_dict)

        state_dict = model.sanitize(state_dict)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

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
