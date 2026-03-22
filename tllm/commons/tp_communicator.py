from abc import ABC

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
else:
    raise ImportError("MLX backend not found")
