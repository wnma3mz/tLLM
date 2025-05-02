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


from dataclasses import dataclass


@dataclass
class NetworkConfig:
    primary_host: str = "localhost"
    primary_port: int = 29500
    timeout: float = 30.0
    retry_attempts: int = 3
    buffer_size: int = 8192
    use_quantization: bool = True
    quantization_bits: int = 8
    tcp_links: int = 4
    metal_buffer_pool_size: int = 1024  # MB
    metal_queue_depth: int = 3


if BACKEND == BackendEnum.MLX:
    import mlx.core as mx
    from mpi4py import MPI

    class MLXCommunicator(ABCCommunicator):
        def __init__(self, logger) -> None:
            super().__init__(logger)
            self.config = NetworkConfig()
            self._configure_metal()
            comm = mx.distributed.init()
            self.world_size = comm.size()
            self.rank = comm.rank()
            if self.world_size > 1:
                MPI.Info.Set("btl_tcp_links", "4")

        def all_reduce(self, x: mx.array) -> mx.array:
            return mx.distributed.all_sum(x)

        def _configure_metal(self):
            """Configure Metal-specific optimizations"""
            # Enable Metal buffer pooling
            os.environ["MLX_BUFFER_POOL_SIZE"] = str(self.config.metal_buffer_pool_size)
            # Set Metal command queue depth
            os.environ["MLX_METAL_QUEUE_DEPTH"] = str(self.config.metal_queue_depth)
            # Enable Metal graph optimization
            os.environ["MLX_METAL_GRAPH_OPTIMIZE"] = "1"
            os.environ["OMPI_MCA_btl_tcp_links"] = str(self.config.tcp_links)

    # Communicator = BaseCommunicator
    Communicator = MLXCommunicator
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
