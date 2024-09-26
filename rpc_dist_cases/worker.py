import os
import time

from common import MyModel, remote_func
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.multiprocessing as mp

# python3 rpc_dist_cases/worker.py
# 会同时启动 4 个进程

ip = "localhost"
port = 22336
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = str(port)
init_method = f"tcp://{ip}:{port}"
dist_init_method = f"tcp://{ip}:{port+1}"


# 定义模型管理类
class ModelManager:
    def __init__(self):
        self.model = MyModel()  # 初始化模型

    def init_input(self, x):
        self.x = x

    @torch.no_grad()
    def forward(self):
        return self.model(self.x)  # 执行模型的前向传播


def run_worker(rank, world_size):
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    if rank == 2:
        print(f"rank {rank} worker{rank}")
        rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
    elif rank <= 1:
        tp_world_size = 2
        dist.init_process_group("gloo", init_method=dist_init_method, rank=rank, world_size=tp_world_size)
        trainer_name = f"trainer{rank}"
        print(f"rank {rank} {trainer_name}")
        rpc.init_rpc(name=trainer_name, rank=rank, world_size=world_size, rpc_backend_options=options)
    else:
        return

    print(f"Worker worker{rank} is ready")
    # print(f"Worker worker{rank} is ready rank{dist.get_rank()}")
    rpc.shutdown()
    # dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
