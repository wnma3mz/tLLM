import time
from typing import *

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.commons.layers import MyLlamaDecoderLayer
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.protocol import ForwardResult, SeqInput
from tllm.models.utils import build_mask
from tllm.rpc.manager import RPCManager
from tllm.rpc.schemas_pb2 import BFloat16Tensor


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    position_ids_list, actual_seq_len_list = [], []
    for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            position_ids = torch.tensor([cache_seq_len], dtype=torch.long).unsqueeze(0)
            actual_seq_len_list.append([seq_len, cache_seq_len + 1])
        else:
            layer_cache_list = None
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            actual_seq_len_list.append([seq_len, seq_len])
        request_cache.add(uuid, seq_len, layer_cache_list)
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
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(dtype)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)

    @classmethod
    def from_pretrained(cls, logger, config, tok: TokenizerUtils, weight_path: str, server: RPCManager):
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

        cls.server = server
        cls.pp_size = server.size
        if tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(tok.tokenizer.eos_token_id)
        eos_token = tok.tokenizer.convert_ids_to_tokens(list(cls.eos_token_ids))
        cls.logger.debug(f"eos_token_ids: {cls.eos_token_ids}; Tokens: {eos_token}")

        state_dict = torch.load(weight_path)
        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.eval()
        return model

    def _prepare_forward_data(
        self, seq_input: SeqInput, hidden_states: torch.Tensor, need_serialize: bool, need_compress: bool
    ) -> Dict[str, Union[List[str], List[int], BFloat16Tensor]]:
        if need_serialize:
            hidden_states = serialize_tensor(hidden_states, need_compress=need_compress)
        return {"uuid": seq_input.uuid_list, "seq_len": seq_input.seq_len_list, "hidden_states": hidden_states}

    def forward(self, inputs_embeds: torch.Tensor, seq_input: SeqInput) -> ForwardResult:
        hidden_states = inputs_embeds
        comm_cost_time_list = []
        need_compress = True if sum(seq_input.seq_len_list) >= 64 else False  # 超过这个数，需要进行压缩传输
        for pp_idx in range(self.pp_size):
            s1 = time.time()
            response = self.server.post_sync(
                pp_idx,
                "/forward",
                data=self._prepare_forward_data(seq_input, hidden_states, pp_idx == 0, need_compress),
            )
            hidden_states = (
                deserialize_tensor(response.output, to_tensor=True, has_compress=need_compress)
                if pp_idx == self.pp_size - 1
                else response.output
            )
            s2 = time.time()
            comm_cost_time_list.append(s2 - s1 - response.cost_time)

        hidden_states = hidden_states.to(self.norm.weight.device)
        # hidden_states/logits: bsz x seq_len x hidden_size
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # bsz: 1; seq_len: seq_len1 + seq_len2
        return ForwardResult(logits=logits, comm_cost_time_list=comm_cost_time_list)
