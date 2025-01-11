import os

from tllm import BACKEND, BackendEnum
from tllm.schemas import MIX_TENSOR


class BaseCommunicator:
    def __init__(self, logger) -> None:
        self.rank = 0
        self.world_size = 1
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
        return x


if BACKEND == BackendEnum.MLX:
    import mlx.core as mx
    from mpi4py import MPI
    import numpy as np

    mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
    MPI._typedict["e"] = mpi_float16

    def sum_f16_cb(inbuf, outbuf, t):
        """Custom reduction function for float16 sum."""
        assert t == mpi_float16  # Ensure data type consistency
        array_a = np.frombuffer(inbuf, dtype="float16")
        array_b = np.frombuffer(outbuf, dtype="float16")
        array_b += array_a
        return array_b  # Return the updated `outbuf`

    # Create the custom reduction operation (commutative for efficiency)
    mpi_sum_f16 = MPI.Op.Create(sum_f16_cb, commute=True)

    class MPICommunicator(BaseCommunicator):
        def __init__(self, logger) -> None:
            self.comm = MPI.COMM_WORLD
            self.world_size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.logger = logger

        def all_reduce(self, x: mx.array) -> mx.array:
            ori_dtype = x.dtype
            x = x.astype(mx.float16)
            self.comm.Allreduce(MPI.IN_PLACE, [x, mpi_float16], op=mpi_sum_f16)
            return x.astype(ori_dtype)

    class MLXCommunicator(BaseCommunicator):
        def __init__(self, logger) -> None:
            comm = mx.distributed.init()
            self.world_size = comm.size()
            self.rank = comm.rank()
            self.logger = logger


        def all_reduce(self, x: mx.array) -> mx.array:
            return mx.distributed.all_sum(x)

    # Feature: MLXCommunicator
    # Communicator = MPICommunicator
    Communicator = MLXCommunicator
elif BACKEND == BackendEnum.TORCH:
    import torch
    import torch.distributed as dist

    class TorchCommunicator(BaseCommunicator):
        def __init__(self, logger, init_method=None, rank=None, world_size=None, is_torchrun: bool = False) -> None:
            if init_method is not None:
                dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
            if is_torchrun:
                dist.init_process_group("gloo")

            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.logger = logger

        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x


    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        Communicator = TorchCommunicator  # is_torchrun=True
    else:
        Communicator = BaseCommunicator
else:
    raise ImportError("No backend found")
