import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from http_comm.server import Server
from utils import tensor_to_list


class TensorParallelLlamaMLP(nn.Module):
    def __init__(self, config, server: Server, layer_idx: int, tp_size: int, offset: int):
        super().__init__()
        self.tp_size = tp_size
        self.layer_idx = layer_idx
        self.offset = offset
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.slice_intermediate_size = self.intermediate_size // self.tp_size
        self.server = server
        self.config = config
        self.load_model_flag = False
        self.act_fn = ACT2FN[config.hidden_act]

    def _post_init(self, state_dict_path: str):
        if self.load_model_flag:
            return
        proj_key_list = ["gate_proj", "up_proj", "down_proj"]
        proj_requests_dict = {}
        for tp_idx in range(self.tp_size):
            proj_list = []
            for proj_key in proj_key_list:
                # weight_data = proj_dict[proj_key]
                base_proj_config = {
                    "mlp_bias": self.config.mlp_bias,
                    "proj_name": proj_key,
                    "tp_idx": tp_idx,
                    "tp_size": self.tp_size,
                    "layer_idx": self.layer_idx + self.offset,
                    "state_dict_path": state_dict_path,
                    # "weight_data": tensor_to_list(weight_data[tp_idx]),
                }
                proj_config = copy.deepcopy(base_proj_config)
                if proj_key == "gate_proj" or proj_key == "up_proj":
                    proj_config.update({"input_size": self.hidden_size, "output_size": self.slice_intermediate_size})
                elif proj_key == "down_proj":
                    proj_config.update({"input_size": self.slice_intermediate_size, "output_size": self.hidden_size})
                else:
                    raise ValueError(f"Invalid proj_key: {proj_key}")
                proj_list.append(proj_config)
            proj_requests_dict[tp_idx] = proj_list
        response_dict = self.server.post_thread_url_dict("/init_mlp", proj_requests_dict)
        for tp_idx in range(self.tp_size):
            for response in response_dict[tp_idx]:
                assert response.status == 200
        self.load_model_flag = True

    def _prepare_forward_data(self, proj_name: str, tp_idx: int, hidden_states: torch.Tensor) -> Dict:
        return {
            "proj_name": proj_name,
            "tp_idx": tp_idx,
            "layer_idx": self.layer_idx + self.offset,
            "hidden_states": tensor_to_list(hidden_states),
        }

    def forward(self, x):
        # send data to each tensor parallel
        gate_proj_list, up_proj_list = [], []
        proj_requests_dict = {}
        for tp_idx in range(self.tp_size):
            proj_data_list = [
                self._prepare_forward_data(proj_name, tp_idx, x) for proj_name in ["gate_proj", "up_proj"]
            ]
            proj_requests_dict[tp_idx] = proj_data_list
        response_dict = self.server.post_thread_url_dict("/forward_mlp", proj_requests_dict)
        for tp_idx in range(self.tp_size):
            gate_proj_slice, up_proj_slice = self.server.fetch_list_output(response_dict[tp_idx])
            gate_proj_list.append(torch.tensor(gate_proj_slice, dtype=x.dtype).to(x.device))
            up_proj_list.append(torch.tensor(up_proj_slice, dtype=x.dtype).to(x.device))
        # concat data
        concat_gate_proj_out = torch.cat(gate_proj_list, dim=-1)
        concat_up_proj_out = torch.cat(up_proj_list, dim=-1)

        intermediate_states = self.act_fn(concat_gate_proj_out) * concat_up_proj_out
        intermediate_states_list = intermediate_states.split(self.slice_intermediate_size, dim=2)

        x_list = [
            self._prepare_forward_data("down_proj", tp_idx, intermediate_states_list[tp_idx])
            for tp_idx in range(self.tp_size)
        ]
        down_proj_slice_list = self.server.post_thread("/forward_mlp", x_list)
        down_proj = sum(
            map(
                lambda out: torch.tensor(out, dtype=x.dtype).to(x.device),
                self.server.fetch_list_output(down_proj_slice_list),
            )
        )
        return down_proj


class TensorParallelLlamaSdpaAttention(nn.Module):
    def __init__(self, config, server: Server, layer_idx: int, tp_size: int, offset: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.tp_size = tp_size
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.tp_size
        self.query_slices = (self.num_heads * self.head_dim) // self.tp_size
        self.o_slicing = self.hidden_size // self.tp_size
        self.server = server
        self.load_model_flag = False
        self.offset = offset

    def _post_init(self, state_dict_path: str):
        if self.load_model_flag:
            return
        proj_key_list = ["q_proj", "k_proj", "v_proj", "o_proj"]
        proj_requests_dict = {}
        for tp_idx in range(self.tp_size):
            proj_config_list = []
            for proj_name in proj_key_list:
                # name = f"{proj_name}_proj_{tp_idx}_layer_idx_{self.layer_idx}"
                # weight_data = tensor_to_list(proj_dict[proj_name][tp_idx])
                if proj_name[0] == "q":
                    input_size, output_size = self.hidden_size, self.query_slices
                elif proj_name[0] == "k" or proj_name[0] == "v":
                    input_size, output_size = self.hidden_size, self.key_value_slicing
                elif proj_name[0] == "o":
                    input_size, output_size = self.o_slicing, self.hidden_size
                else:
                    raise ValueError(f"Invalid proj_name: {proj_name}")
                proj_config = {
                    "input_size": input_size,
                    "output_size": output_size,
                    "tp_idx": tp_idx,
                    "tp_size": self.tp_size,
                    "layer_idx": self.layer_idx + +self.offset,
                    "mlp_bias": self.config.mlp_bias,
                    "proj_name": proj_name,
                    "state_dict_path": state_dict_path,
                    # "weight_data": weight_data,
                }
                proj_config_list.append(proj_config)
            proj_requests_dict[tp_idx] = proj_config_list
        response_dict = self.server.post_thread_url_dict("/init_mlp", proj_requests_dict)
        for tp_idx in range(self.tp_size):
            for response in response_dict[tp_idx]:
                assert response.status == 200
        self.load_model_flag = True

    def _prepare_forward_data(self, proj_name: str, tp_idx: int, hidden_states: torch.Tensor) -> Dict:
        return {
            "proj_name": proj_name,
            "tp_idx": tp_idx,
            "layer_idx": self.layer_idx + self.offset,
            "hidden_states": tensor_to_list(hidden_states),
        }

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # send data to each tensor parallel
        query_list, key_list, value_list = [], [], []
        proj_requests_dict = {}
        for tp_idx in range(self.tp_size):
            data_list = []
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                data = self._prepare_forward_data(proj_name, tp_idx, tensor_to_list(hidden_states))
                data_list.append(data)
            proj_requests_dict[tp_idx] = data_list

        proj_response_dict = self.server.post_thread_url_dict("/forward_mlp", proj_requests_dict)
        for tp_idx in range(self.tp_size):
            query, key, value = self.server.fetch_list_output(proj_response_dict[tp_idx])
            query_list.append(torch.tensor(query, dtype=hidden_states.dtype).to(hidden_states.device))
            key_list.append(torch.tensor(key, dtype=hidden_states.dtype).to(hidden_states.device))
            value_list.append(torch.tensor(value, dtype=hidden_states.dtype).to(hidden_states.device))

        # concat data
        query_states = torch.cat(query_list, dim=-1)
        key_states = torch.cat(key_list, dim=-1)
        value_states = torch.cat(value_list, dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = attn_output.split(self.o_slicing, dim=2)
        data_list = [
            self._prepare_forward_data("o_proj", tp_idx, tensor_to_list(attn_output[tp_idx]))
            for tp_idx in range(self.tp_size)
        ]
        attn_output_list = self.server.post_thread("/forward_mlp", data_list)

        attn_output = sum(
            map(
                lambda out: torch.tensor(out, dtype=hidden_states.dtype).to(hidden_states.device),
                self.server.fetch_list_output(attn_output_list),
            )
        )
        return attn_output, None, past_key_value


class TensorParallelLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, server: Server, layer_idx: int, tp_size: int, offset):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = TensorParallelLlamaSdpaAttention(config, server, layer_idx, tp_size, offset)

        self.mlp = TensorParallelLlamaMLP(config, server, layer_idx, tp_size, offset)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
