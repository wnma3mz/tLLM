import argparse
import time
from typing import *

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from transformers import AutoConfig, LlamaForCausalLM

from src3.commons.communicator import Communicator
from src3.model import MyLlamaModel


class ModelManager:
    def init_model(self, start_layer_idx: int, end_layer_idx: int, model_path: str):
        """
        @param start_layer_idx: 开始层数
        @param end_layer_idx: 结束层数
        @param model_path: 模型路径
        """
        device = "cpu"
        comm = Communicator()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.decoder_start_layer_idx = start_layer_idx
        config.decoder_end_layer_idx = end_layer_idx
        config.comm = comm

        print(f"[Rank: {config.comm.rank}] World Size: {config.comm.world_size}")

        s1 = time.time()
        state_dict = LlamaForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map=device
        ).state_dict()
        model = MyLlamaModel(config)
        model.load_state_dict(state_dict)
        print(f"[Rank: {config.comm.rank}] Load Model Cost Time {time.time() - s1}")
        model.to(device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, uuid_str: str) -> torch.Tensor:
        return self.model(hidden_states, uuid_str=uuid_str)  # 执行模型的前向传播


def run_worker(rank: int, world_size: int, pp_range: Tuple[int], init_method: str, dist_init_method: str):
    """
    @param rank: 当前进程的 rank
    @param world_size: 当前进程的 world size
    @param pp_range: 当前进程的 pp 范围
    """
    if rank < pp_range[0] or rank > pp_range[1]:
        return
    tp_world_size = pp_range[1] - pp_range[0] + 1
    tp_rank = rank - pp_range[0]  # 对应节点使用的 rank id

    worker_name = f"worker{rank}"
    print(f"[RPC Rank: {rank}, [TP Rank: {tp_rank}]] {worker_name};")

    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    options.rpc_timeout = 180000

    # 每个 RPC 进程对应一个 distributed 的进程
    dist.init_process_group("gloo", init_method=dist_init_method, rank=tp_rank, world_size=tp_world_size)
    rpc.init_rpc(name=worker_name, rank=rank, world_size=world_size, rpc_backend_options=options)
    rpc.shutdown()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=3, help="world size")
    parser.add_argument("--pp_start", type=int, default=1, help="pp start")
    parser.add_argument("--pp_end", type=int, default=2, help="pp end")
    parser.add_argument("--ip", type=str, default="localhost", help="ip")
    parser.add_argument("--rpc_port", type=int, default=29605, help="rpc port")
    parser.add_argument("--process_port", type=int, default=29607, help="process port")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_method = f"tcp://{args.ip}:{args.rpc_port}"
    dist_init_method = f"tcp://{args.ip}:{args.process_port}"

    world_size = args.world_size
    pp_range = (args.pp_start, args.pp_end)
    mp.spawn(run_worker, args=(world_size, pp_range, init_method, dist_init_method), nprocs=world_size, join=True)
