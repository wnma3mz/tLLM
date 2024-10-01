from typing import *

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from tllm.commons.cache_manager import CacheManager
from tllm.commons.layers import MyLlamaDecoderLayer


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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        next_decoder_cache = None
        for i, layer in enumerate(self.decoder):
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

            # 所有层的 kv cache 放到一起了，所以这里只需要取最后一层的 kv cache
            next_decoder_cache = layer_outputs[1]
        next_cache = next_decoder_cache
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


class MyLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.decoder = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx)

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decoder.load_state_dict(state_dict)

    def forward(self, hidden_states: torch.Tensor, uuid_str: Optional[str] = None) -> torch.Tensor:
        """
        @param hidden_states: bs x seq_len x hidden_size
        @param position_ids: bs x seq_len
        @param uuid_str: 可选的 uuid，用于区分不同的请求。如果 uuid_str 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: bs x seq_len x hidden_size
        """
        if uuid_str in self.cache_manager.cache_dict:
            kv_cache_seq_len = self.cache_manager.cache_dict[uuid_str]["past_key_values"].get_seq_length()
            position_ids = torch.tensor([kv_cache_seq_len], dtype=torch.long).unsqueeze(0)
            past_key_values = self.cache_manager.get(uuid_str)["past_key_values"]
        else:
            position_ids = torch.arange(hidden_states.size(1), dtype=torch.long).unsqueeze(0)
            past_key_values = DynamicCache()
        hidden_states = hidden_states.to(self.device)
        position_ids = position_ids.to(self.device)
        output = self.decoder(hidden_states, position_ids=position_ids, past_key_value=past_key_values)

        self.cache_manager.set(uuid_str, output.past_key_values)
        self.cache_manager.check_alive()
        return output.last_hidden_state

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
