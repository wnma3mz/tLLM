import time

import torch
import torch.distributed as dist
import torch.nn as nn

# Run command
# torchrun --nproc_per_node=2 linear_cases/parallel/dist_torch.py


def setup_seed(seed):
    torch.manual_seed(seed)


class MyModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ColumnParallelLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert hidden_size % self.world_size == 0
        self.layer = nn.Linear(hidden_size, hidden_size // self.world_size, bias=False)

    def load_weight(self, w: torch.Tensor):
        w = w.chunk(self.world_size, dim=0)[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        cluster_output = (
            [torch.zeros_like(node_output, dtype=node_output.dtype) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.gather(node_output, gather_list=cluster_output, dst=0)
        return torch.cat(cluster_output, dim=-1) if self.rank == 0 else None

    def forward_all_gather(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        cluster_output = [torch.zeros_like(node_output, dtype=node_output.dtype) for _ in range(self.world_size)]
        dist.all_gather(cluster_output, node_output)
        return torch.cat(cluster_output, dim=-1) if self.rank == 0 else None


class RowParallelLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert hidden_size % self.world_size == 0
        self.layer = nn.Linear(hidden_size // self.world_size, hidden_size, bias=False)

    def load_weight(self, w: torch.Tensor):
        w = w.chunk(self.world_size, dim=1)[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_x = torch.chunk(x, self.world_size, dim=-1)[self.rank]
        node_output = self.layer(node_x)
        dist.all_reduce(node_output, op=dist.ReduceOp.SUM)
        return node_output if self.rank == 0 else None


def broadcast_func(x):
    y = torch.zeros_like(x)
    if dist.get_rank() == 0:
        y = x + 1
    dist.broadcast(y, src=0)
    print(f"Rank: {dist.get_rank()}; value: {y}")


if __name__ == "__main__":
    setup_seed(42)
    # 初始化分布式环境
    dist.init_process_group(backend="gloo")

    # x = torch.tensor([1,2])
    # broadcast_func(x)

    # 示例输入和权重
    hidden_size = 4096
    iters = 10
    input_tensor = torch.randn(1, 2, hidden_size)  # 假设输入
    weight_tensor = torch.randn(hidden_size, hidden_size)  # 假设权重

    # 调用推理
    parallel_model = RowParallelLayer(hidden_size)
    # parallel_model = ColumnParallelLayer(hidden_size)
    parallel_model.load_weight(weight_tensor)

    dist_output = parallel_model(input_tensor)

    dist_cost_time_list, base_cost_time_list = [], []
    for _ in range(iters):
        s1 = time.time()
        parallel_model(input_tensor)
        dist_cost_time = time.time() - s1
        dist_cost_time_list.append(dist_cost_time)
    if dist.get_rank() == 0:
        print(f"rank: {dist.get_rank()}; shape: {dist_output.shape}")

        model = MyModel(hidden_size)
        model.load_state_dict(dict({"layer.weight": weight_tensor}))
        base_output = model(input_tensor)

        for _ in range(iters):
            s1 = time.time()
            model(input_tensor)
            # base_inference(input_tensor, weight_tensor)
            base_cost_time_list.append(time.time() - s1)

        print("world size", dist.get_world_size())
        print("base cost time", sum(base_cost_time_list) / len(base_cost_time_list))
        print("dist cost time", sum(dist_cost_time_list) / len(dist_cost_time_list))

        is_same = torch.allclose(dist_output, base_output)
        print("is same", is_same)
        if not is_same:
            print("abs diff：", abs(dist_output - base_output).sum())
