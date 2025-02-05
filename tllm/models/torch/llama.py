from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm import DEVICE, DTYPE
from tllm.models.torch.helper import TorchCacheManager
from tllm.models.torch.layers import Decoder
from tllm.models.weight_helper import default_merge_attn, default_merge_mlp, tie_word_embeddings_func
from tllm.schemas import SeqInput

cache_manager = TorchCacheManager()


class HFLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)[0]
        position_ids_expanded = position_ids[None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HFLlamaModel(nn.Module):
    def __init__(self, config, is_merge: bool = True):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        self.rotary_emb = HFLlamaRotaryEmbedding(config=config)

        max_seq_len = getattr(self.model.layers[-1].self_attn, "max_seq_len", -1)
        num_key_value_heads = self.model.layers[-1].self_attn.num_key_value_heads
        head_dim = self.model.layers[-1].self_attn.head_dim
        cache_manager.init_request_cache(self.num_decoder_layers, max_seq_len, num_key_value_heads, head_dim)

        is_start_pp = self.config.decoder_start_layer_idx == 0
        is_end_pp = self.config.decoder_end_layer_idx == self.config.num_hidden_layers
        cache_manager.post_init(is_start_pp, is_end_pp)

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, torch.Tensor], is_merge: bool = True, **kwargs):
        model = cls(config, is_merge)
        state_dict = model.merge_weights(state_dict, is_merge)
        state_dict = model.sanitize(state_dict, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        model.load_state_dict(state_dict)

        model.to(DTYPE).to(DEVICE)
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

    def merge_weights(self, state_dict: Dict[str, torch.Tensor], is_merge: bool) -> Dict[str, torch.Tensor]:
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

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor, seq_input: SeqInput) -> torch.Tensor:
        """
        @param hidden_states: seq_len x hidden_size
        @param seq_input:
            uuid_list: List[str]: 每个请求的 uuid
            seq_len_list: List[int]: 每个请求的 seq_len
            如果 uuid 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: seq_len x hidden_size
        """
        hidden_states = cache_manager.build_forward_cache(hidden_states, seq_input)

        position_embeddings = self.rotary_emb(hidden_states, cache_manager.position_ids)

        hidden_states = self.model(
            hidden_states, position_embeddings=position_embeddings, attention_data=cache_manager.attn_data
        )

        # TODO 异步更新 cache
        cache_manager.update_cache(seq_input)

        output = cache_manager.get_last_hidden_states(hidden_states)
        return output


class HFLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=DEVICE)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=DEVICE)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Optional[Dict] = None, **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        state_dict = tie_word_embeddings_func(config, state_dict)
        state_dict = model.sanitize(state_dict)
        model.load_state_dict(state_dict)
        model.to(DTYPE).to(DEVICE)
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

    @torch.inference_mode()
    def get_input_embeddings(self, x: np.ndarray) -> torch.Tensor:
        return self.embed_tokens(torch.tensor(x, device=DEVICE))

    @torch.inference_mode()
    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(DTYPE).to(DEVICE)
        # (seq_len1+seq_len2) x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        return logits
