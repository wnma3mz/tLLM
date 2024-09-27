import time
from typing import *

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from transformers import AutoConfig, LlamaForCausalLM

from src3.commons.communicator import Communicator
from src3.model import MyLlamaModel

ip = "localhost"
port = 29605
init_method = f"tcp://{ip}:{port}"
dist_init_method = f"tcp://{ip}:{port+1}"


def get_current_rank():
    # 获取当前工作者的信息
    worker_info = rpc.get_worker_info()
    if worker_info is not None:
        return worker_info.id  # 返回当前工作者的 ID（rank）
    else:
        raise RuntimeError("This function can only be called in an RPC worker.")


class ModelManager:
    def init_model(self):
        comm = Communicator()
        model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.decoder_start_layer_idx = config.num_hidden_layers // 2
        config.decoder_end_layer_idx = config.num_hidden_layers
        config.comm = comm

        print(f"[Rank: {config.comm.rank}] World Size: {config.comm.world_size}")

        s1 = time.time()
        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()
        model = MyLlamaModel(config)
        model.load_state_dict(state_dict)
        print(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
        model.to("cpu")
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, hidden_states, uuid_str: str):
        out = self.model(hidden_states, uuid_str=uuid_str)  # 执行模型的前向传播
        return self.model._prepare_output_data(uuid_str, out)


def run_worker(rank, world_size, start_rank, end_rank):
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    options.rpc_timeout = 180000
    if rank in range(start_rank, end_rank + 1):
        worker_name = f"worker{rank}"
        print(f"[Rank: {rank}] {worker_name}")
        tp_world_size = end_rank - start_rank + 1
        dist.init_process_group("gloo", init_method=init_method, rank=rank - start_rank, world_size=tp_world_size)
        rpc.init_rpc(name=worker_name, rank=rank, world_size=world_size, rpc_backend_options=options)

        rpc.shutdown()
        dist.destroy_process_group()
    else:
        return


if __name__ == "__main__":
    world_size = 4
    start_rank, end_rank = 0, 1
    mp.spawn(
        run_worker,
        args=(
            world_size,
            start_rank,
            end_rank,
        ),
        nprocs=world_size,
        join=True,
    )
