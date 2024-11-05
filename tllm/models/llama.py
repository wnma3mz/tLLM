from typing import *

import numpy as np
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm.commons.layers import MyLlamaDecoderLayer
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.protocol import SeqInput
from tllm.models.utils import build_mask, load_master_weight


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    position_ids_list, actual_seq_len_list = [], []
    for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            # kv_cache 是整个历史的 kv_cache
            # 当 q_len 为 1 时，直接使用 kv_cache，使用历史的全部 token kv cache
            # TODO: 当 q_len > 1 时，表示只需要使用前 q_len 的 kv_cache，后面的 kv_cache 需要重新计算
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            position_ids = torch.tensor([cache_seq_len], dtype=torch.long).unsqueeze(0)
            actual_seq_len_list.append([q_len, cache_seq_len + q_len])  # q_len 是需要新计算 kv_cache 的长度
        else:
            layer_cache_list = None
            position_ids = torch.arange(q_len, dtype=torch.long).unsqueeze(0)
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
    def __init__(self, config, start_layer_idx: int, end_layer_idx: int):
        super().__init__()
        config.offset = start_layer_idx
        self.decoder = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(start_layer_idx, end_layer_idx)]
        )

    def load_state_dict(self, state_dict: Dict) -> None:
        for layer in self.decoder:
            layer.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data: AttentionData,
    ) -> torch.Tensor:
        for layer in self.decoder:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_data=attention_data)
        return hidden_states


class MyLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.decoder = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decoder.load_state_dict(state_dict)

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
        hidden_states = self.decoder(
            hidden_states, position_embeddings=position_embeddings, attention_data=attention_data
        )

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()
        return hidden_states

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(self.dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(self.dtype)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(self.dtype)

    @classmethod
    def from_pretrained(cls, logger, config, tok: TokenizerUtils, model_path: str):
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

        if tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(tok.tokenizer.eos_token_id)
        eos_token = tok.tokenizer.convert_ids_to_tokens(list(cls.eos_token_ids))
        cls.logger.debug(f"eos_token_ids: {cls.eos_token_ids}; Tokens: {eos_token}")

        state_dict = load_master_weight(model_path)
        embedding_weight = state_dict.pop("model.embed_tokens.weight")
        model.embed_tokens.load_state_dict({"weight": embedding_weight})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        if "lm_head.weight" in state_dict:
            model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})
        else:
            model.lm_head.load_state_dict({"weight": model.embed_tokens.weight})

        model.eval()
        return model

    @torch.no_grad()
    def get_input_embeddings(self, x: np.ndarray) -> torch.Tensor:
        return self.embed_tokens(torch.tensor(x))

    @torch.no_grad()
    def get_logits(self, hidden_states: torch.Tensor, seq_len_list: List[int]) -> torch.Tensor:
        # 只取最后一个 token 的 hidden_states
        seq_hidden_states = torch.split(hidden_states, [seq_len for seq_len in seq_len_list], dim=0)
        hidden_states = torch.cat([x[-1:, :] for x in seq_hidden_states], dim=0)
        hidden_states = hidden_states.to(self.dtype).to(self.norm.weight.device)
        # bsz x seq_len x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        # bsz: 1; seq_len: seq_len1 + seq_len2
        return logits
