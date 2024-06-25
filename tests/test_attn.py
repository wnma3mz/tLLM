from dataclasses import dataclass

import torch
from server import Server

from models.llama.layers import TensorParallelLlamaSdpaAttention


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    mlp_bias: bool
    attention_dropout: float = 0.0
    head_dim: int = 2
    num_key_value_heads: int = 2
    num_attention_heads: int = 2
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    hidden_act: str = "gelu"


if __name__ == "__main__":
    url_list = ["http://localhost:8000", "http://localhost:8001"]
    server = Server(url_list)
    hidden_size, intermediate_size = 2, 4
    config = Config(hidden_size=hidden_size, intermediate_size=intermediate_size, mlp_bias=False)
    t_mlp = TensorParallelLlamaSdpaAttention(config, server, layer_idx=0, tp_size=2)

    state_dict = {
        "q_proj.weight": torch.randn(config.num_attention_heads * config.head_dim, hidden_size),
        "k_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, hidden_size),
        "v_proj.weight": torch.randn(config.num_key_value_heads * config.head_dim, hidden_size),
        "o_proj.weight": torch.randn(hidden_size, hidden_size),
    }
    t_mlp._post_init(state_dict)
    # attention_mask = torch.tensor([[0, 1], [1, 0]])
    position_ids = torch.LongTensor([[0, 1]])
    t_mlp.forward(torch.randn(1, 2, 2), position_ids=position_ids)
