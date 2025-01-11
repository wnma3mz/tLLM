import os
from typing import Any, List

from tllm import BACKEND, BackendEnum
from tllm.schemas import MIX_TENSOR


class BaseCommunicator:
    def __init__(self, **kwargs) -> None:
        self.rank = 0
        self.world_size = 1

    def is_rank0(self) -> bool:
        return self.rank == 0

    def print_rank0(self, *args):
        if self.is_rank0():
            print(*args)

    def all_reduce(self, x: MIX_TENSOR) -> MIX_TENSOR:
        return x

    def all_gather(self, x: MIX_TENSOR):
        return x

    def gather(self, x: MIX_TENSOR):
        return x

    def broadcast(self, x: MIX_TENSOR):
        return x

    def broadcast_object(self, x: List[Any]):
        return x


if BACKEND == BackendEnum.MLX:
    import mlx.core as mx
    from mpi4py import MPI

    class MLXCommunicator(BaseCommunicator):
        def __init__(self) -> None:
            self.comm = MPI.COMM_WORLD
            self.world_size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

        def all_reduce(self, x: mx.array) -> mx.array:
            # input shape == output shape
            x = x.astype(mx.float32)
            global_out = mx.zeros_like(x)
            self.comm.Allreduce(memoryview(x), memoryview(global_out), op=MPI.SUM)
            return global_out.astype(mx.float16)

        def all_gather(self, x: mx.array):
            raise NotImplementedError

        def gather(self, x: mx.array):
            raise NotImplementedError

        def broadcast(self, x: mx.array):
            raise NotImplementedError

        def broadcast_object(self, obj_list: Any):
            return self.comm.bcast(obj_list, root=0)

    # Feature: MLXCommunicator
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        print("Using MLXCommunicator")
        Communicator = MLXCommunicator
    else:
        Communicator = BaseCommunicator
elif BACKEND == BackendEnum.TORCH:
    import torch
    import torch.distributed as dist

    class TorchCommunicator(BaseCommunicator):
        def __init__(self, init_method=None, rank=None, world_size=None, is_torchrun: bool = False) -> None:
            if init_method is not None:
                dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
            if is_torchrun:
                dist.init_process_group("gloo")

            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

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

        def broadcast_object(self, obj_list: List[Any]):
            # self.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
            # self.comm.broadcast(hidden_states)
            dist.broadcast_object_list(obj_list, src=0)

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        Communicator = TorchCommunicator  # is_torchrun=True
    else:
        Communicator = BaseCommunicator
else:
    raise ImportError("No backend found")
