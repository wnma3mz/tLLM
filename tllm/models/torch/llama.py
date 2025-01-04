from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm import DEVICE, DTYPE
from tllm.commons.attn import get_attention_implementation
from tllm.commons.cache import CacheManager
from tllm.models.torch.helper import build_forward_cache, get_last_hidden_states
from tllm.models.torch.layers import Decoder
from tllm.models.utils import read_eos_token_ids
from tllm.models.weight_helper import default_merge_attn, default_merge_mlp
from tllm.schemas import SeqInput

_, attention_type = get_attention_implementation()


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
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        self.rotary_emb = HFLlamaRotaryEmbedding(config=config)

    @classmethod
    def from_pretrained(cls, config, state_dict: Dict[str, torch.Tensor], is_merge: bool = True, **kwargs):
        model = cls(config, is_merge)
        state_dict = model.merge_weights(state_dict, is_merge)
        model.load_state_dict(state_dict)

        model.to(DTYPE).to(DEVICE)
        model.eval()
        return model

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
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_decoder_layers)
        hidden_states = hidden_states.to(DTYPE).to(DEVICE)
        position_embeddings = self.rotary_emb(hidden_states, attention_data.position_ids.to(DEVICE))
        if attention_type == "flash_attention":
            attention_data.attn_mask = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in attention_data.attn_mask.items()
            }
        else:
            attention_data.attn_mask = attention_data.attn_mask.to(DEVICE)

        hidden_states = self.model(
            hidden_states, position_embeddings=position_embeddings, attention_data=attention_data
        )

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            hidden_states = get_last_hidden_states(hidden_states, seq_input.seq_len_list)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()

        return hidden_states


class HFLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, state_dict: Optional[Dict] = None, **kwargs):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.eos_token_ids = read_eos_token_ids(config)

        model.load_state_dict(state_dict)
        model.to(DTYPE).to(DEVICE)
        model.eval()
        return model

    @torch.inference_mode()
    def get_input_embeddings(self, x: np.ndarray) -> torch.Tensor:
        return self.embed_tokens(torch.tensor(x, device=DEVICE))

    @torch.inference_mode()
    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(DTYPE).to(DEVICE)
        # (seq_len1+seq_len2) x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        return logits
