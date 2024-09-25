from typing import *

import torch
import torch.distributed as dist


class Communicator:
    def __init__(self) -> None:
        dist.init_process_group(backend="gloo")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def is_rank0(self):
        return dist.get_rank() == 0

    def print_rank0(self, *args):
        if self.is_rank0():
            print(*args)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # input shape == output shape
        # output = torch.sum(torch.stack(input), dim=0)
        # each node get output
        # with self.lock:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor):
        cluster_output = [torch.zeros_like(x, dtype=x.dtype) for _ in range(self.world_size)]
        dist.all_gather(cluster_output, x)
        return torch.cat(cluster_output, dim=-1)

    def gather(self, x: torch.Tensor):
        # 只在节点 0 上聚合
        cluster_output = (
            [torch.zeros_like(x, dtype=x.dtype) for _ in range(self.world_size)] if self.rank == 0 else None
        )
        dist.gather(x, gather_list=cluster_output, dst=0)
        return torch.cat(cluster_output, dim=-1) if self.rank == 0 else None

    def broadcast(self, x: torch.Tensor):
        dist.broadcast(x, src=0)
