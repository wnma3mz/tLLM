import re
from typing import *

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaRotaryEmbedding

from tllm.commons.layers import LlamaDecoderLayer
from tllm.models.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.torch_helper import build_mask
from tllm.models.utils import load_master_weight
from tllm.schemas import SeqInput


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    position_ids_list, actual_seq_len_list = [], []
    for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            # kv_cache 是整个历史的 kv_cache
            # 当 q_len 为 1 时，直接使用 kv_cache，使用历史的全部 token kv cache
            # TODO: 当 q_len > 1 时，表示只需要使用前 q_len 的 kv_cache，后面的 kv_cache 需要重新计算
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            position_ids = torch.tensor([cache_seq_len], dtype=torch.long)
            actual_seq_len_list.append([q_len, cache_seq_len + q_len])  # q_len 是需要新计算 kv_cache 的长度
        else:
            layer_cache_list = None
            position_ids = torch.arange(q_len, dtype=torch.long)
            actual_seq_len_list.append([q_len, q_len])
        request_cache.add(uuid, q_len, layer_cache_list)
        position_ids_list.append(position_ids)
    return AttentionData(
        request_cache=request_cache,
        attn_mask=build_mask(actual_seq_len_list),
        uuid_list=seq_input.uuid_list,
        position_ids=torch.cat(position_ids_list, dim=-1),
    )


class Decoder(nn.Module):
    def __init__(self, config, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        super().__init__()
        config.offset = start_layer_idx
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx, is_merge) for layer_idx in range(start_layer_idx, end_layer_idx)]
        )

    @torch.no_grad()
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
    @torch.no_grad()
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


class LlamaModel(nn.Module):
    def __init__(self, config, is_merge: bool = True):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        self.rotary_emb = TLlamaRotaryEmbedding(config=config)

    @classmethod
    def from_pretrained(cls, config, model_path: str, state_dict: Optional[Any] = None, is_merge: bool = True):
        model = cls(config, is_merge)
        state_dict = LlamaForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map="cpu", torch_dtype=model.dtype, low_cpu_mem_usage=True
        ).state_dict()
        state_dict = model.read_weight_from_model_path(state_dict, is_merge)
        model.load_state_dict(state_dict)
        del state_dict

        model.to(model.dtype).to(model.device)
        model.eval()
        return model

    def read_weight_from_model_path(self, weights: Dict[str, torch.Tensor], is_merge: bool) -> Dict[str, torch.Tensor]:
        # TODO: support bias and TP

        attn_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn")
        mlp_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.mlp")
        layer_name_mapper = {
            "self_attn.o_proj": "self_attn.o_proj.layer",
            "mlp.down_proj": "mlp.down_proj.layer",
        }
        qkv_proj_list = ["q_proj", "k_proj", "v_proj"]
        gate_up_list = ["gate_proj", "up_proj"]

        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
        key_list = list(weights.keys())
        for key in key_list:
            for prefix_key in prefix_key_list:
                if key.startswith(prefix_key):
                    weights.pop(key)
        if not is_merge:
            return weights

        key_list = list(weights.keys())

        attn_proj_w = {}  # layer_idx -> {qkv: weight}
        mlp_w = {}
        for key in key_list:
            for s_key, t_key in layer_name_mapper.items():
                if s_key in key:
                    # w_list = w.chunk(self.world_size, dim=1)[self.rank]
                    weights[key.replace(s_key, t_key)] = weights.pop(key)
            attn_res = attn_layer_idx_pattern.findall(key)
            mlp_res = mlp_layer_idx_pattern.findall(key)
            if attn_res:
                layer_idx = int(attn_res[0])
                if layer_idx not in attn_proj_w:
                    attn_proj_w[layer_idx] = {}
            elif mlp_res:
                layer_idx = int(mlp_res[0])
                if layer_idx not in mlp_w:
                    mlp_w[layer_idx] = {}
            else:
                continue

            for qkv in qkv_proj_list:
                if qkv in key:
                    attn_proj_w[layer_idx].update({qkv: weights.pop(key)})
            for mlp in gate_up_list:
                if mlp in key:
                    mlp_w[layer_idx].update({mlp: weights.pop(key)})

            layer_weights = attn_proj_w.get(layer_idx, [])
            if len(layer_weights) == 3:
                name = f"model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
                # torch.chunk(layer_weights[qkv], self.world_size, dim=0)
                weights[name] = torch.cat([layer_weights[qkv] for qkv in qkv_proj_list], dim=0)
                attn_proj_w.pop(layer_idx)

            layer_weights = mlp_w.get(layer_idx, [])
            if len(layer_weights) == 2:
                name = f"model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
                # torch.chunk(layer_weights[mlp], self.world_size, dim=0)
                weights[name] = torch.cat([layer_weights[mlp] for mlp in gate_up_list], dim=0)
                mlp_w.pop(layer_idx)

        return weights

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, seq_input: SeqInput) -> torch.Tensor:
        """
        @param hidden_states: bs x seq_len x hidden_size
        @param seq_input:
            uuid_list: List[str]: 每个请求的 uuid
            seq_len_list: List[int]: 每个请求的 seq_len
            如果 uuid 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: bs x seq_len x hidden_size
        """
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_decoder_layers)
        hidden_states = hidden_states.to(self.device)
        position_embeddings = self.rotary_emb(hidden_states, attention_data.position_ids.to(self.device))
        attention_data.attn_mask = attention_data.attn_mask.to(self.device)
        hidden_states = self.model(
            hidden_states, position_embeddings=position_embeddings, attention_data=attention_data
        )

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()
        return hidden_states


class TLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        state_dict = load_master_weight(model_path)
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

    @torch.no_grad()
    def get_input_embeddings(self, x: np.ndarray) -> torch.Tensor:
        return self.embed_tokens(torch.tensor(x, device=self.device))

    @torch.no_grad()
    def get_logits(self, hidden_states: torch.Tensor, seq_len_list: List[int]) -> List[torch.Tensor]:
        # 只取最后一个 token 的 hidden_states
        seq_hidden_states = torch.split(hidden_states, [seq_len for seq_len in seq_len_list], dim=0)
        hidden_states = torch.cat([x[-1:, :] for x in seq_hidden_states], dim=0)
        hidden_states = hidden_states.to(self.dtype).to(self.norm.weight.device)
        # seq_len x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        # seq_len -> seq_len1 + seq_len2
        return torch.chunk(logits, len(seq_len_list), dim=0)
