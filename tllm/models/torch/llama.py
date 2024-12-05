import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm.commons.attn import get_attention_implementation
from tllm.commons.cache import AttentionData, CacheManager
from tllm.models.torch.helper import EmptyLayer, build_forward_cache, read_from_safetensors
from tllm.models.torch.layers import LlamaDecoderLayer
from tllm.models.utils import (
    default_merge_attn_weight,
    default_merge_mlp_weight,
    get_model_path,
    get_weight_path,
    pop_weight_func,
    read_eos_token_ids,
)
from tllm.schemas import SeqInput

_, attention_type = get_attention_implementation()

DTYPE = torch.bfloat16
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = "cpu"


class Decoder(nn.Module):
    def __init__(self, config, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        super().__init__()
        config.offset = start_layer_idx
        self.layers = nn.ModuleList(
            [EmptyLayer()] * start_layer_idx
            + [LlamaDecoderLayer(config, layer_idx, is_merge) for layer_idx in range(start_layer_idx, end_layer_idx)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data: AttentionData,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_data=attention_data)
        return hidden_states


class TLlamaRotaryEmbedding(LlamaRotaryEmbedding):
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


def get_last_hidden_states(hidden_states: torch.Tensor, seq_len_list: List[int]) -> torch.Tensor:
    # 只取最后一个 token 的 hidden_states
    seq_hidden_states = torch.split(hidden_states, [seq_len for seq_len in seq_len_list], dim=0)
    return torch.cat([x[-1:, :] for x in seq_hidden_states], dim=0)


class HFLlamaModel(nn.Module):
    def __init__(self, config, is_merge: bool = True):
        super().__init__()
        self.dtype = DTYPE
        self.device = DEVICE
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        self.rotary_emb = TLlamaRotaryEmbedding(config=config)

    @classmethod
    def from_pretrained(cls, config, model_path: str, state_dict: Optional[Dict] = None, is_merge: bool = True):
        model = cls(config, is_merge)
        model_path = get_model_path(model_path)
        state_dict = model.read_weight_from_model_path(model_path, is_merge)
        model.load_state_dict(state_dict)

        model.to(model.dtype).to(model.device)
        model.eval()
        return model

    def read_weight_from_model_path(self, model_path: str, is_merge: bool) -> Dict[str, torch.Tensor]:
        # TODO: support bias and TP
        weights = {}
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))
        for file in weight_files:
            weights.update(read_from_safetensors(os.path.join(model_path, file)))

        layer_name_mapper = {
            "self_attn.o_proj": "self_attn.o_proj.layer",
            "mlp.down_proj": "mlp.down_proj.layer",
        }
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
        weights = pop_weight_func(
            prefix_key_list,
            weights,
            self.config.num_hidden_layers,
            self.config.decoder_start_layer_idx,
            self.config.decoder_end_layer_idx,
        )

        key_list = list(weights.keys())
        for key in key_list:
            for s_key, t_key in layer_name_mapper.items():
                if s_key in key:
                    # w_list = w.chunk(self.world_size, dim=1)[self.rank]
                    weights[key.replace(s_key, t_key)] = weights.pop(key)

        if not is_merge:
            return weights
        # torch.chunk(layer_weights[qkv], self.world_size, dim=0)
        weights = default_merge_attn_weight(weights)
        # torch.chunk(layer_weights[mlp], self.world_size, dim=0)
        weights = default_merge_mlp_weight(weights)
        return weights

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
        hidden_states = hidden_states.to(self.device)
        position_embeddings = self.rotary_emb(hidden_states, attention_data.position_ids.to(self.device))
        if attention_type == "flash_attention":
            attention_data.attn_mask = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in attention_data.attn_mask.items()
            }
        else:
            attention_data.attn_mask = attention_data.attn_mask.to(self.device)

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
        self.dtype = DTYPE
        self.device = DEVICE
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, model_path: str, state_dict: Optional[Dict] = None):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.eos_token_ids = read_eos_token_ids(config)

        model_path = get_model_path(model_path)
        file_set, prefix_key_list = get_weight_path(model_path)
        state_dict = {}
        for file in file_set:
            weight_path = os.path.join(model_path, file)
            state_dict.update(read_from_safetensors(weight_path, prefix_key_list))

        state_dict = {k.split("model.")[-1]: v for k, v in state_dict.items()}
        has_key_list = list(state_dict.keys())
        if "lm_head.weight" not in state_dict:
            for key in has_key_list:
                if key.startswith("embed_tokens."):
                    state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]

        model.load_state_dict(state_dict)
        model.to(model.dtype).to(model.device)
        model.eval()
        return model

    @torch.inference_mode()
    def get_input_embeddings(self, x: np.ndarray) -> torch.Tensor:
        return self.embed_tokens(torch.tensor(x, device=self.device))

    @torch.inference_mode()
    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype).to(self.norm.weight.device)
        # (seq_len1+seq_len2) x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        return logits
