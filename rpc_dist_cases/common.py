import torch.distributed as dist
import torch.nn as nn


def remote_func(x):
    print(f"[Rank {dist.get_rank()}] [World Size {dist.get_world_size()}] {x}")
    return x


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        print("x", x)
        return self.layer(x)
