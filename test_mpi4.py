import os
import socket

import mlx.core as mx
from mpi4py import MPI

from tllm.commons.communicator import MLXCommunicator

hostname = socket.gethostname()
comm = MLXCommunicator()

a = mx.ones((42, 4096))
a_out = comm.all_reduce(a)
print(a_out.shape)
# world = mx.distributed.init()
# x = mx.distributed.all_sum(mx.ones(10))
# print(f"Distributed available: {mx.distributed.is_available()}")
# print(f"Hostname: {hostname}: {world.rank()}, {x}")

# attn_output (42, 2048)
