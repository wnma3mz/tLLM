from typing import *

import mlx.core as mx
import mlx.nn as nn

from tllm.models.mlx.layers import MergeParallelLayer


def setup_seed(seed: int):
    mx.random.seed(seed)


class MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        return self.layer1(x), self.layer2(x)


if __name__ == "__main__":
    setup_seed(0)

    x = mx.random.uniform(0, 10, (2, 4))
    hidden_size = 4
    mlp = MLP(hidden_size)
    out = mlp(x)

    concat_w = mx.concat([mlp.layer1.weight, mlp.layer2.weight], axis=0)

    merge_mlp = MergeParallelLayer(hidden_size, hidden_size, 2, 1, 0)
    merge_mlp.load_weight([concat_w])
    merge_out = merge_mlp(x)

    assert mx.allclose(out[0], merge_out[0])
    assert mx.allclose(out[1], merge_out[1])
