from typing import Dict, List

import torch
import torch.nn as nn

from schemas import MLPConfig, MLPForwardData
from utils import tensor_to_list


class MLP(nn.Module):
    def __init__(self, data: MLPConfig):
        super().__init__()
        self.mlp = nn.Linear(data.input_size, data.output_size, bias=data.mlp_bias)
        if data.weight_data:
            self.mlp.load_state_dict(
                {
                    "weight": torch.tensor(data.weight_data),
                    "bias": torch.tensor(data.bias_data),
                }
            )
        elif data.state_dict_path:
            state_dict = torch.load(data.state_dict_path, "cpu")
            if data.proj_name in ["gate_proj", "up_proj", "down_proj"]:
                layer_name = (
                    f"model.layers.{data.layer_idx}.mlp.{data.proj_name}.weight"
                )
                chunk_dim = 1 if data.proj_name == "down_proj" else 0
            elif data.proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                layer_name = (
                    f"model.layers.{data.layer_idx}.self_attn.{data.proj_name}.weight"
                )
                chunk_dim = 1 if data.proj_name[0] == "o" else 0
            else:
                raise ValueError(f"Invalid proj_name: {data.proj_name}")

            proj_state_dict = {
                "weight": state_dict[layer_name]
                .chunk(data.tp_size, dim=chunk_dim)[data.tp_idx]
                .clone()
            }
            self.mlp.load_state_dict(proj_state_dict)
        else:
            raise ValueError("Invalid data")

    def _prepare_forward_data(self, data: MLPForwardData) -> torch.Tensor:
        return torch.tensor(data.hidden_states, dtype=self.mlp.weight.dtype)

    def forward(self, x):
        return self.mlp(x)

    def _prepare_output_data(self, hidden_states: torch.tensor) -> List:
        return tensor_to_list(hidden_states)
