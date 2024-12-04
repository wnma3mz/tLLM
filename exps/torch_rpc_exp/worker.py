import argparse
import time
from typing import *

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from transformers import AutoConfig, LlamaForCausalLM

from tllm.commons.communicator import Communicator
from tllm.models.torch.llama import MyLlamaModel
from tllm.schemas import NodeConfig


def get_state_dict(model_path: str, device: str) -> Dict[str, torch.Tensor]:
    state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device).state_dict()
    return state_dict


class ModelManager:
    def init_model(self, node_config: NodeConfig):
        """
        @param node_config: 当前节点的配置
            start_layer_idx: 当前节点开始层
            end_layer_idx: 当前节点结束层
            model_path: 模型路径
            rank: 当前节点的 rank
            prev_rank: 当前节点的前一个节点的 rank
            next_start_rank: 当前节点的下一个节点的开始 rank
            next_end_rank: 当前节点的下一个节点的结束 rank
        """
        device = "cpu"
        comm = Communicator()
        config = AutoConfig.from_pretrained(node_config.checkpoint_path, trust_remote_code=True)
        config.decoder_start_layer_idx = node_config.start_layer_idx
        config.decoder_end_layer_idx = node_config.end_layer_idx
        config.comm = comm

        self.next_rank_list = range(node_config.next_start_rank, node_config.next_end_rank + 1)
        print(f"[Rank: {config.comm.rank}] World Size: {config.comm.world_size}")

        s1 = time.time()
        # TODO: 只在 rank0 获取 state_dict
        state_dict = get_state_dict(node_config.checkpoint_path, device)
        model = MyLlamaModel(config)
        model.load_state_dict(state_dict)
        print(f"[Rank: {config.comm.rank}] Load Model Cost Time {time.time() - s1}")
        model.to(device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, shape_hidden_states: Tuple[int], uuid: str) -> torch.Tensor:
        # 不是 rank0，需要同步 hidden_states
        if hidden_states is None:
            hidden_states = torch.zeros(shape_hidden_states, dtype=self.model.dtype, device=self.model.device)
        self.model.config.comm.broadcast(hidden_states)
        hidden_states = self.model(hidden_states, uuid=uuid)
        return hidden_states


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
