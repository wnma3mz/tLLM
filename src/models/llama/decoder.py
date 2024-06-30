import logging
import os
from dataclasses import dataclass
from typing import *

import numpy as np
from transformers.configuration_utils import PretrainedConfig

from cache_manager import CacheManager
from http_comm.server import Server
from models.common.layers import TransformerBlock, compute_cos_sin_cache

# from models.llama.layers import TensorParallelLlamaDecoderLayer
from schemas import ForwardData, LayerConfig
from utils import tensor_to_list

logging.basicConfig(level=logging.INFO)


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: np.ndarray
    past_key_values: Optional[None]


class DynamicCache:
    def __init__(self) -> None:
        self.key_cache: List[np.ndarray] = []
        self.value_cache: List[np.ndarray] = []
        self._seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: np.ndarray,
        value_states: np.ndarray,
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = np.concatenate([self.key_cache[layer_idx], key_states], axis=1)
            self.value_cache[layer_idx] = np.concatenate([self.value_cache[layer_idx], value_states], axis=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None


class Decoder:
    def post_init(self, config: LayerConfig):
        self.offset = config.layer_idx_start
        self.layer_idx_end = config.layer_idx_end
        self.tp_url_list = config.tp_url_list
        self.tp_size = config.tp_size
        self.layer_state_dict_dir = config.layer_state_dict_dir

        config = PretrainedConfig.from_dict(config.config)
        config._attn_implementation = "sdpa"
        self.config = config
        self.config.torch_dtype = np.float16

        server = None
        if self.tp_size > 1:
            server = Server(self.tp_url_list)

        layer_list = []
        for i, layer_idx in enumerate(range(self.offset, self.layer_idx_end)):
            print(f"offload layer idx: {layer_idx}")
            if self.tp_size > 1:
                layer = ...
                # layer = self._init_layer_tp(config, layer_idx, i, server)
            else:
                layer = self._init_layer_general(config, layer_idx, i)
            layer_list.append(layer)

        self.decoder = layer_list
        self.cache_manager = CacheManager()
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(
            config.hidden_size // config.num_attention_heads, config.max_position_embeddings
        )

        # self.cache_manager.cache_dict[uuid]["past_key_values"]
        # key_cache/value_cache: (layer_idx, batch_size, num_heads, seq_len, head_dim) for transformer
        # key_cache/value_cache: (layer_idx, batch_size, seq_len, num_heads, head_dim) for numpy

    # def _init_layer_tp(self, config: PretrainedConfig, layer_idx: int, i: int, server: Server):
    #     layer = TensorParallelLlamaDecoderLayer(
    #         config, server=server, layer_idx=i, tp_size=self.tp_size, offset=self.offset
    #     )

    #     layer_state_dict_path = os.path.join(self.layer_state_dict_dir, f"layer_{layer_idx}.pth")
    #     layer_state_dict = np.load(layer_state_dict_path)
    #     layer_state_dict = {k.split(f"model.layers.{layer_idx}.")[-1]: v for k, v in layer_state_dict.items()}
    #     layer.self_attn._post_init(layer_state_dict_path)
    #     if not layer.self_attn.load_model_flag:
    #         raise ValueError("load self attention model failed")
    #     layer.mlp._post_init(layer_state_dict_path)
    #     if not layer.mlp.load_model_flag:
    #         raise ValueError("load mlp model failed")
    #     layer.input_layernorm.load_state_dict({"weight": layer_state_dict[f"input_layernorm.weight"]})
    #     layer.post_attention_layernorm.load_state_dict({"weight": layer_state_dict[f"post_attention_layernorm.weight"]})
    #     return layer

    def _init_layer_general(self, config: PretrainedConfig, layer_idx: int, i: int):
        layer = TransformerBlock(config, layer_idx=i)

        layer_state_dict = np.load(
            os.path.join(self.layer_state_dict_dir, f"layer_{layer_idx}.npy"), allow_pickle=True
        ).item()
        layer_state_dict = {k.split(f"model.layers.{layer_idx}.")[-1]: v for k, v in layer_state_dict.items()}
        layer.load_state_dict(layer_state_dict)
        return layer

    def _prepare_forward_data(self, data: ForwardData) -> np.ndarray:
        hidden_states = np.array(data.hidden_states, dtype=self.config.torch_dtype)
        # 客户端自行生成 position_ids
        if data.uuid in self.cache_manager.cache_dict:
            kv_cache_seq_len = (
                self.cache_manager.cache_dict[data.uuid]["past_key_values"].key_cache[0].shape[1]
            )  # different from the transformer
            request_seq_len = hidden_states.shape[1] - 1
            assert kv_cache_seq_len == request_seq_len, "seq_len not match"
            start_pos = request_seq_len
            past_key_values = self.cache_manager.get(data.uuid)["past_key_values"]
        else:
            # 首次请求，没有 kv cache
            start_pos = 0
            past_key_values = DynamicCache()

        return {
            "hidden_states": hidden_states,
            "start_pos": start_pos,
            "past_key_values": past_key_values,
        }

    def forward(
        self,
        hidden_states: np.ndarray,
        start_pos: int,  # 第几个 token
        past_key_values=None,
    ):
        # 默认 use_cache=True, 存储 kv cache
        _, seq_len, _ = hidden_states.shape  # [B, L, D]
        mask = np.full((seq_len, seq_len), float("-inf"))
        mask = np.triu(mask, k=1)
        mask = np.concatenate([np.zeros((seq_len, start_pos)), mask], axis=1)
        # 如果是第一个 token， freqs shape 为 0:seq_len，否则为 start_post:start_pos+1
        freqs_cos = self.freqs_cos[start_pos:seq_len]
        freqs_sin = self.freqs_sin[start_pos:seq_len]

        next_decoder_cache = None
        for layer in self.decoder:
            layer_outputs = layer(
                hidden_states,
                start_pos=start_pos,
                mask=mask,
                freqs_cos_sin=(freqs_cos, freqs_sin),
                past_key_value=past_key_values,
            )
            hidden_states = layer_outputs[0]

            # 所有层的 kv cache 放到一起了，所以这里只需要取最后一层的 kv cache
            next_decoder_cache = layer_outputs[1]
        next_cache = next_decoder_cache
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)

    def _prepare_output_data(self, data: ForwardData, output: BaseModelOutputWithPast) -> List:
        self.cache_manager.set(data.uuid, output.past_key_values)
        self.cache_manager.check_alive()
        return tensor_to_list(output.last_hidden_state)
