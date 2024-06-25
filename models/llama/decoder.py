import os
from typing import *

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from cache_manager import CacheManager
from models.llama.layers import TensorParallelLlamaDecoderLayer
from server import Server
from tllm_schemas import ForwardData, LayerConfig
from utils import tensor_to_list


class Decoder:
    def post_init(self, config: LayerConfig):
        self.offset = config.layer_idx_start
        self.layer_idx_end = config.layer_idx_end
        self.tp_url_list = config.tp_url_list
        self.tp_size = config.tp_size
        state_dict_path = config.state_dict_path
        # layer_state_dict_path = config.layer_state_dict_path

        config = PretrainedConfig.from_dict(config.config)
        config._attn_implementation = "sdpa"

        server = None
        if self.tp_size >= 1:
            server = Server(self.tp_url_list)

        state_dict = torch.load(state_dict_path, "cpu")
        layer_list = []
        for i, layer_idx in enumerate(range(self.offset, self.layer_idx_end)):
            print(f"offload layer idx: {layer_idx}")
            if self.tp_size >= 1:
                layer = self._init_layer_tp(config, layer_idx, i, server)
            else:
                layer = self._init_layer_general(config, layer_idx, i, state_dict)
            layer_list.append(layer)

        self.decoder = nn.ModuleList(layer_list)
        self.cache_manager = CacheManager()
        # self.cache_manager.cache_dict[uuid]["past_key_values"]
        # key_cache/value_cache: (layer_idx, batch_size, num_heads, seq_len, head_dim)

    def _init_layer_tp(self, config: PretrainedConfig, layer_idx: int, i: int, server: Server) -> nn.Module:
        layer = TensorParallelLlamaDecoderLayer(
            config, server=server, layer_idx=i, tp_size=self.tp_size, offset=self.offset
        )

        state_dict_path = f"./weights/TinyLlama-1.1B-chat-v1.0-pp1_layer_{layer_idx}.pth"
        layer_state_dict = torch.load(state_dict_path, "cpu")
        layer_state_dict = {k.split(f"model.layers.{layer_idx}.")[-1]: v for k, v in layer_state_dict.items()}
        layer.self_attn._post_init(state_dict_path)
        if not layer.self_attn.load_model_flag:
            raise ValueError("load self attention model failed")
        layer.mlp._post_init(state_dict_path)
        if not layer.mlp.load_model_flag:
            raise ValueError("load mlp model failed")
        layer.input_layernorm.load_state_dict({"weight": layer_state_dict[f"input_layernorm.weight"]})
        layer.post_attention_layernorm.load_state_dict({"weight": layer_state_dict[f"post_attention_layernorm.weight"]})
        return layer

    def _init_layer_general(self, config: PretrainedConfig, layer_idx: int, i: int, state_dict: Dict) -> nn.Module:
        layer = LlamaDecoderLayer(config, layer_idx=i)

        layer_state_dict = {}
        for k, v in state_dict.items():
            prefix_str = f"model.layers.{layer_idx}."
            if prefix_str in k:
                layer_state_dict[k.split(prefix_str)[-1]] = v
        layer.load_state_dict(layer_state_dict)
        return layer

    def _prepare_forward_data(self, data: ForwardData) -> torch.Tensor:
        hidden_states = torch.tensor(data.hidden_states)
        # 客户端自行生成 position_ids
        if data.uuid in self.cache_manager.cache_dict:
            kv_cache_seq_len = self.cache_manager.cache_dict[data.uuid]["past_key_values"].key_cache[0].shape[-2]
            request_seq_len = hidden_states.size(1) - 1
            assert kv_cache_seq_len == request_seq_len, "seq_len not match"
            position_ids = torch.tensor([request_seq_len], dtype=torch.long).unsqueeze(0)
            past_key_values = self.cache_manager.get(data.uuid)["past_key_values"]
        else:
            position_ids = torch.arange(hidden_states.size(1), dtype=torch.long).unsqueeze(0)
            past_key_values = DynamicCache()

        return {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        # 默认 use_cache=True, 存储 kv cache
        next_decoder_cache = None
        for layer in self.decoder:
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]

            # 所有层的 kv cache 放到一起了，所以这里只需要取最后一层的 kv cache
            next_decoder_cache = layer_outputs[1]
        next_cache = next_decoder_cache
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)

    def _prepare_output_data(self, data: ForwardData, output: BaseModelOutputWithPast) -> Dict[str, Any]:
        self.cache_manager.set(data.uuid, output.past_key_values)
        self.cache_manager.check_alive()
        return {
            "last_hidden_state": tensor_to_list(output.last_hidden_state),
        }


if __name__ == "__main__":
    import pickle

    from utils import list_to_tensor, tensor_to_list

    with open("inputs_embeds.pkl", "rb") as f:
        inputs_embeds = tensor_to_list(pickle.load(f))
    with open("pkv.pkl", "rb") as f:
        pp_past_key_values = pickle.load(f)
    past_key_values = pp_past_key_values[0]

    data = ForwardData(hidden_states=inputs_embeds, key_cache=past_key_values[0], value_cache=past_key_values[1])

    from transformers import AutoConfig

    model_name = "TinyLlama-1.1B-Chat-v1.0"
    model_path = f"/Users/lujianghu/Documents/{model_name}"
    config = AutoConfig.from_pretrained(model_path)

    model = Decoder()
    model.post_init(config, 0, 22)
    # pipeline_parallel = 1
    # fname = f"./weights/{model_name}-pp{pipeline_parallel}/decoder_pp0.pth"
    # model.load_state_dict(torch.load(fname, "cpu"))

    input_data = model._prepare_forward_data(data)

    # assert False
    output = model.forward(**input_data)
    print(output)
    # output = model.forward(hidden_states)
