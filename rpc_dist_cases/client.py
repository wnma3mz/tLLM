import os
import time

from common import remote_func
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from worker import ModelManager


def call_remote_init(model_rref, input_tensor):
    return model_rref.rpc_sync().init_input(input_tensor)


def call_remote_forward(model_rref):
    return model_rref.rpc_sync().forward()


# python3 rpc_dist_cases/client.py
# 会启动 1 个进程

ip = "localhost"
port = 22336
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = str(port)

init_method = f"tcp://{ip}:{port}"


def run_client(rank, world_size):
    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    rpc.init_rpc(name=f"client{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)

    x = torch.tensor([1, 2, 3])
    fut = rpc.rpc_async("trainer1", remote_func, args=(x,))
    result = fut.wait()
    print(f"result 1: {result}")

    x = torch.rand((1, 10))
    # 通过RPC从worker获取RRef（这里假设我们调用worker0的模型）
    model_rref = rpc.remote("worker2", ModelManager)
    # 调用远程worker的forward函数
    # fut = rpc.rpc_sync("worker2", inference, args=(x, ))
    # result = fut.wait()
    result = call_remote_init(model_rref, x)
    result = call_remote_forward(model_rref)
    print(f"result 2: {result}")

    rpc.shutdown()


if __name__ == "__main__":
    processes = []
    world_size = 4
    mp.set_start_method("spawn")
    p = mp.Process(target=run_client, args=(3, world_size))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
