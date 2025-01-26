from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm import DTYPE
from tllm.commons.cache import CacheManager
from tllm.models.mlx.helper import build_forward_cache, get_last_hidden_states, quantization_func
from tllm.models.mlx.layers import Decoder
from tllm.models.weight_helper import default_merge_attn, default_merge_mlp
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

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        attention_data = build_forward_cache(
            seq_input, self.cache_manager, self.num_layers, self.max_seq_len, self.n_kv_heads, self.head_dim
        )

        mask = attention_data.attn_mask
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = self.model(hidden_states, mask=mask, cache=attention_data)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            output = get_last_hidden_states(output, seq_input.seq_len_list)
        return output

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], **kwargs):
        is_merge = True

        model = cls(config, is_merge)
        state_dict = model.merge_weights(state_dict, is_merge)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

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

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))  # , strict=False

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states.astype(DTYPE)))
        return logits
