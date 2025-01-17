from abc import ABC
import os

from tllm import BACKEND, BackendEnum
from tllm.schemas import MIX_TENSOR


class ABCCommunicator(ABC):
    def __init__(self, logger) -> None:
        self.logger = logger

    def is_rank0(self) -> bool:
        return self.rank == 0

    def info_rank0(self, *args):
        if self.is_rank0():
            self.logger.info(*args)

    def debug_rank0(self, *args):
        if self.is_rank0():
            self.logger.debug(*args)

    def all_reduce(self, x: MIX_TENSOR) -> MIX_TENSOR:
        raise NotImplementedError


class BaseCommunicator(ABCCommunicator):
    def __init__(self, logger) -> None:
        super().__init__(logger)
        self.rank = 0
        self.world_size = 1

    def all_reduce(self, x: MIX_TENSOR) -> MIX_TENSOR:
        return x


if BACKEND == BackendEnum.MLX:
    import mlx.core as mx

    class MLXCommunicator(ABCCommunicator):
        def __init__(self, logger) -> None:
            super().__init__(logger)
            comm = mx.distributed.init()
            self.world_size = comm.size()
            self.rank = comm.rank()

        def all_reduce(self, x: mx.array) -> mx.array:
            return mx.distributed.all_sum(x)

    Communicator = BaseCommunicator
elif BACKEND == BackendEnum.TORCH:
    import torch
    import torch.distributed as dist

    class TorchCommunicator(ABCCommunicator):
        def __init__(self, logger, init_method=None, rank=None, world_size=None, is_torchrun: bool = False) -> None:
            super().__init__(logger)
            if init_method is not None:
                dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
            if is_torchrun:
                dist.init_process_group("gloo")

            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        Communicator = TorchCommunicator  # is_torchrun=True
    else:
        Communicator = BaseCommunicator
else:
    raise ImportError("No backend found")
