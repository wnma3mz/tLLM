from dataclasses import dataclass

import torch
import torch.nn as nn

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.models.torch.layers import MyLlamaMLP


@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    comm: SingleNodeCommunicator
    attention_dropout: float = 0.0
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling = None


def setup_seed():
    torch.manual_seed(0)


def test_linear():
    hidden_size = 4
    mlp = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype)

    bsz, seq_len, hidden_size = 1, 2, hidden_size
    input_1 = torch.randn((bsz, seq_len, hidden_size), dtype=dtype)
    input_2 = torch.randn((bsz, seq_len, hidden_size), dtype=dtype)

    output_1 = mlp(input_1)
    output_2 = mlp(input_2)

    continue_batch = torch.cat([input_1, input_2], dim=1)
    output_continue_batch = mlp(continue_batch)

    cat_output = torch.cat([output_1, output_2], dim=1)
    print(f"is_same: {torch.allclose(cat_output, output_continue_batch)}")


def test_mlp():
    mlp = MyLlamaMLP(config, 0)
    mlp.to(dtype)

    bsz, seq_len, hidden_size = 1, 2, config.hidden_size
    input_1 = torch.randn((bsz, seq_len, hidden_size), dtype=dtype)
    input_2 = torch.randn((bsz, seq_len, hidden_size), dtype=dtype)

    output_1 = mlp(input_1)
    output_2 = mlp(input_2)

    continue_batch = torch.cat([input_1, input_2], dim=1)
    output_continue_batch = mlp(continue_batch)

    cat_output = torch.cat([output_1, output_2], dim=1)
    print(f"is_same: {torch.allclose(cat_output, output_continue_batch)}")


if __name__ == "__main__":
    setup_seed()
    dtype = torch.bfloat16
    comm = SingleNodeCommunicator()

    config = Config(hidden_size=4, intermediate_size=16, hidden_act="silu", comm=comm)
