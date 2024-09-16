from dataclasses import dataclass

import torch
from server import Server

from models.llama.layers import TensorParallelLlamaMLP


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    mlp_bias: bool
    hidden_act: str = "gelu"


if __name__ == "__main__":
    url_list = ["http://localhost:8000", "http://localhost:8001"]
    server = Server(url_list)
    hidden_size, intermediate_size = 2, 8
    config = Config(
        hidden_size=hidden_size, intermediate_size=intermediate_size, mlp_bias=False
    )
    t_mlp = TensorParallelLlamaMLP(config, server, layer_idx=0, tp_size=2)

    state_dict = {
        "gate_proj.weight": torch.randn(intermediate_size, hidden_size),
        "up_proj.weight": torch.randn(intermediate_size, hidden_size),
        "down_proj.weight": torch.randn(hidden_size, intermediate_size),
    }
    t_mlp._post_init(state_dict)
    t_mlp.forward(torch.randn(1, 2, 2))
