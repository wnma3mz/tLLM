import os
import socket
from typing import List

import mlx.core as mx
import numpy as np

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


class SocketCommunicator(BaseCommunicator):
    def __init__(self, logger, world_size: int, rank: int, hosts: List[str], ports: List[int]):
        self.world_size = world_size
        self.rank = rank
        self.sockets: List[socket.socket] = []
        self.packet_alignment = 64  # Align to cache line size
        self.buffer_size = 1024 * 1024  # 1MB chunks for efficient transfer
        self.logger = logger

        # Initialize socket library
        if hasattr(socket, "TCP_NODELAY"):
            self.TCP_NODELAY = socket.TCP_NODELAY
        if hasattr(socket, "TCP_QUICKACK"):
            self.TCP_QUICKACK = socket.TCP_QUICKACK

        if rank == 0:
            # Root node: create server and accept connections
            self.server = self._create_server_socket(hosts[0], ports[0])
            self.sockets = self._accept_connections()
        else:
            # Worker nodes: connect to root and other nodes
            self.sockets = self._connect_to_peers(hosts, ports)

        print("len(self.sockets):", len(self.sockets))

    def _create_server_socket(self, host: str, port: int) -> socket.socket:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_TCP, self.TCP_NODELAY, 1)
        if hasattr(socket, "TCP_QUICKACK"):
            server.setsockopt(socket.IPPROTO_TCP, self.TCP_QUICKACK, 1)
        server.bind((host, port))
        server.listen(self.world_size)
        return server

    def _accept_connections(self) -> List[socket.socket]:
        sockets = []
        for _ in range(self.world_size - 1):
            client, _ = self.server.accept()
            client.setsockopt(socket.IPPROTO_TCP, self.TCP_NODELAY, 1)
            if hasattr(socket, "TCP_QUICKACK"):
                client.setsockopt(socket.IPPROTO_TCP, self.TCP_QUICKACK, 1)
            sockets.append(client)
        return sockets

    def _connect_to_peers(self, hosts: List[str], ports: List[int]) -> List[socket.socket]:
        sockets = []
        for i in range(self.world_size):
            if i != self.rank:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, self.TCP_NODELAY, 1)
                if hasattr(socket, "TCP_QUICKACK"):
                    s.setsockopt(socket.IPPROTO_TCP, self.TCP_QUICKACK, 1)
                s.connect((hosts[i], ports[i]))
                sockets.append(s)
        return sockets

    def _aligned_buffer(self, size: int) -> memoryview:
        """Create an aligned buffer for efficient transfer"""
        extra = size % self.packet_alignment
        if extra:
            size += self.packet_alignment - extra
        return memoryview(bytearray(size))

    def _send_buffer(self, sock: socket.socket, data: memoryview):
        """Send data in efficient chunks"""
        total_sent = 0
        size = len(data)
        while total_sent < size:
            chunk_size = min(self.buffer_size, size - total_sent)
            sent = sock.send(data[total_sent : total_sent + chunk_size])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent

    def _recv_buffer(self, sock: socket.socket, size: int) -> memoryview:
        """Receive data in efficient chunks"""
        buffer = self._aligned_buffer(size)
        total_received = 0
        while total_received < size:
            chunk_size = min(self.buffer_size, size - total_received)
            chunk = sock.recv(chunk_size)
            if not chunk:
                raise RuntimeError("Socket connection broken")
            buffer[total_received : total_received + len(chunk)] = chunk
            total_received += len(chunk)
        return buffer

    def all_reduce(self, x: mx.array) -> mx.array:
        """Perform all-reduce operation using ring algorithm"""
        dtype = x.dtype
        x_np = x.astype(mx.float16)  # .numpy()
        size = x_np.nbytes

        # Ring algorithm
        send_buff = memoryview(x_np)
        recv_buff = self._aligned_buffer(size)

        for i in range(self.world_size - 1):
            # Send to next rank
            # next_rank = (self.rank + 1) % self.world_size
            next_rank = 0
            self._send_buffer(self.sockets[next_rank], send_buff)

            # Receive from previous rank
            # prev_rank = (self.rank - 1) % self.world_size
            prev_rank = 0
            recv_data = self._recv_buffer(self.sockets[prev_rank], size)

            # Reduce
            recv_array = np.frombuffer(recv_data, dtype=np.float16).reshape(x_np.shape)
            x_np += mx.array(recv_array, dtype=mx.float16)

            # Update send buffer for next iteration
            send_buff = memoryview(x_np)

        return x_np.astype(dtype)

    def close(self):
        """Clean up resources"""
        for sock in self.sockets:
            sock.close()
        if hasattr(self, "server"):
            self.server.close()


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
    Communicator = SocketCommunicator
    # Communicator = MLXCommunicator
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
