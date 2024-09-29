import time
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn

# 合并 多个线性层，并行计算


def setup_seed(seed):
    torch.manual_seed(seed)


class BaseModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.q(x), self.k(x), self.v(x))


class MergeColumn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)

    @torch.no_grad()
    def forward(self, x):
        out = self.qkv(x)
        return torch.chunk(out, 3, dim=-1)


class MergeColumnParallel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert hidden_size % self.world_size == 0
        self.layer = nn.Linear(hidden_size, hidden_size * 3 // self.world_size, bias=False)

    def load_weight(self, w: torch.Tensor):
        w = w.chunk(self.world_size, dim=0)[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        # q, k, v = torch.chunk(node_output, 3, dim=-1)

        cluster_output = (
            [torch.zeros_like(node_output, dtype=node_output.dtype) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.gather(node_output, gather_list=cluster_output, dst=0)
        return torch.chunk(torch.cat(cluster_output, dim=-1), 3, dim=-1) if self.rank == 0 else None


if __name__ == "__main__":
    setup_seed(42)
    dist.init_process_group(backend="gloo")

    hidden_size = 4096
    wq = torch.rand((hidden_size, hidden_size))
    wk = torch.rand((hidden_size, hidden_size))
    wv = torch.rand((hidden_size, hidden_size))

    w_qkv = torch.cat((wq, wk, wv), dim=0)
    x = torch.rand((1, 20, hidden_size))

    base_model = BaseModel(hidden_size)
    merge_column = MergeColumn(hidden_size)
    merge_column_parallel = MergeColumnParallel(hidden_size)
    base_model.load_state_dict({"q.weight": wq, "k.weight": wk, "v.weight": wv})
    merge_column.load_state_dict({"qkv.weight": w_qkv})
    merge_column_parallel.load_weight(w_qkv)

    parallel_out = merge_column_parallel(x)

    # iters = 10
    # parallel_lst = []
    # for _ in range(iters):
    #     s1 = time.time()
    #     merge_column_parallel(x)
    #     parallel_lst.append(time.time() - s1)

    if dist.get_rank() == 0:
        base_q_out, base_k_out, base_v_out = base_model(x)
        q_out, k_out, v_out = merge_column(x)
        parallel_q_out, parallel_k_out, parallel_v_out = parallel_out

        # print(f"parallel cost time: ", sum(parallel_lst) / len(parallel_lst))

        # time_lst = []
        # for _ in range(iters):
        #     s1 = time.time()
        #     base_model(x)
        #     time_lst.append(time.time() - s1)
        # print(f"base_model cost time: ", sum(time_lst) / len(time_lst))

        # time_lst = []
        # for _ in range(iters):
        #     s1 = time.time()
        #     merge_column(x)
        #     time_lst.append(time.time() - s1)
        # print(f"merge_column cost time: ", sum(time_lst) / len(time_lst))

        # print("qkv", q_out, k_out, v_out)
        # print("out", out)
        print(torch.allclose(parallel_q_out, base_q_out))
        print(torch.allclose(parallel_k_out, base_k_out))
        print(torch.allclose(parallel_v_out, base_v_out))

    # print(torch.allclose(base_q_out, q_out))
    # print(torch.allclose(base_k_out, k_out))
    # print(torch.allclose(base_v_out, v_out))
