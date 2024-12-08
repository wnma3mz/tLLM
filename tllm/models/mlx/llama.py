from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm.commons.cache import AttentionData, CacheManager
from tllm.models.mlx.helper import build_forward_cache, empty_func, get_last_hidden_states, quantization_func
from tllm.models.mlx.layers import MLXTransformerBlock
from tllm.models.utils import read_eos_token_ids
from tllm.models.weight_helper import default_merge_attn_weight, default_merge_mlp_weight
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

        # TODO 异步保存 cache
        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            output = get_last_hidden_states(output, seq_input.seq_len_list)
        return output

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], is_merge: bool = True, **kwargs):
        if getattr(config, "quantization", None) is not None or state_dict is not None:
            is_merge = False

        model = cls(config, is_merge)
        state_dict = model.merge_weights(state_dict, is_merge)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()), strict=False)
        mx.eval(model.parameters())
        model.eval()
        return model

    def merge_weights(self, state_dict: Dict[str, mx.array], is_merge: bool = True) -> Dict[str, mx.array]:
        if not is_merge:
            return state_dict
        state_dict = default_merge_attn_weight(state_dict)
        state_dict = default_merge_mlp_weight(state_dict)
        return state_dict


class MLXLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Optional[Any], **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.eos_token_ids = read_eos_token_ids(config)

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array) -> mx.array:
        logits = self.lm_head(self.norm(hidden_states.astype(self.norm.weight.dtype)))
        return logits
