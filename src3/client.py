import argparse
import time
from typing import *
import uuid

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from src3.utils import call_remote_forward, call_remote_init, parse_range_string, setup_seed, tokenize_message
from src3.worker import ModelManager

# 使用 torch.dist 实现 张量并行，使用 torch.dist.rpc 实现 管道并行，通信时仅通信输入


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)
        cls.config = config

        s1 = time.time()
        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()

        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})
        print(f"[Master] Load Model Cost Time {time.time() - s1}")

        model.eval()
        return model

    def forward(self, inputs_embeds: torch.Tensor, uuid_str: str) -> torch.Tensor:
        """
        @param inputs_embeds: bs x seq_len x hidden_size
        @param uuid_str: 用于区分不同请求的 uuid

        @return: bs x seq_len x vocab_size
        """
        hidden_states = inputs_embeds

        for i, model_rref_list in enumerate(self.model_rref_list_list):
            # TODO
            # 每个 node 只发送一次，但需要同步 kv cache 和 hidden states
            # 每个 node 直接发送至下一个 node
            fut_list = []
            for model_rref in model_rref_list:
                fut = call_remote_forward(model_rref, hidden_states, uuid_str)
                fut_list.append(fut)
            hidden_states = fut_list[0].wait()

        hidden_states = torch.tensor(hidden_states).to(inputs_embeds.dtype).to(self.norm.weight.device)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[List[int], Optional[torch.Tensor]]:
        """
        @param input_ids: bs x seq_len
        @param kwargs:
            max_new_tokens: 最大生成 token 数

        @return: token list, 可选的返回内容
        """
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        token_list: List[int] = []
        cnt = 0
        uuid_str = str(uuid.uuid4())
        while True:
            # print(f"Rank: {dist.get_rank()}, input_embeds: {input_embeds}")
            logits = self(input_embeds, uuid_str=uuid_str)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            next_token = torch.argmax(logits[:, -1], dim=1)
            token_list.append(next_token[0].tolist())
            print("token", token_list[-1])
            input_embeds = self.embed_tokens(next_token).unsqueeze(0)

        return token_list, None


def load_model_and_tokenizer(model_path: str) -> Tuple[MyLlamaForCausalLM, AutoTokenizer]:
    model = MyLlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


def run_client(rank: int, world_size: int, model_path: str, pp_ranges: List[Tuple[int, int]], init_method: str):
    if rank != 0:
        return
    # master: embedding + lm head
    # PP: decoder layer 其他部分

    model, tok = load_model_and_tokenizer(model_path)

    # 根据 pp_ranges 计算 start_layer_idx 和 end_layer_idx
    # 假设每个 PP 的层是一样的，所以必须得满足整除
    assert model.config.num_hidden_layers % len(pp_ranges) == 0
    # 计算每个 PP 的层数
    layer_per_pp = model.config.num_hidden_layers // len(pp_ranges)
    # 所以每个 PP 的 start_layer_idx 和 end_layer_idx
    start_layer_idx = 0
    layer_idx_list = []
    for _ in pp_ranges:
        end_layer_idx = start_layer_idx + layer_per_pp
        layer_idx_list.append((start_layer_idx, end_layer_idx))
        start_layer_idx = end_layer_idx
    print(f"layer_idx_list: {layer_idx_list}")

    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    options.rpc_timeout = 180000
    rpc.init_rpc(name=f"client{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)

    # TODO: 优化只给每个 PP 发送一次
    # init and load model
    model_rref_list_list = []
    for (start, end), (start_layer_idx, end_layer_idx) in zip(pp_ranges, layer_idx_list):
        model_rref_list = []
        for remote_rank in range(start, end + 1):
            model_rref = rpc.remote(f"worker{remote_rank}", ModelManager)
            fut = call_remote_init(model_rref, start_layer_idx, end_layer_idx, model_path)
            model_rref_list.append(model_rref)
        s1 = time.time()
        fut.wait()  # 伪分布式需要在 init 后等待，实际上可以在最后一起等待
        print(f"[Client] Init Cost Time: {time.time() - s1}")
        model_rref_list_list.append(model_rref_list)

    # TODO 优化
    model.model_rref_list_list = model_rref_list_list

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print(f"input_ids: {input_ids}")

    s1 = time.time()
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print(f"token: {output[0]}")
    print(f"cost time: {time.time() - s1}")

    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        print(f"Time taken: {time.time() - s1}")
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=5)
    parser.add_argument("--ip", type=str, default="localhost", help="ip")
    parser.add_argument("--rpc_port", type=int, default=29605, help="rpc port")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--pp_ranges", type=parse_range_string, default="1-2,3-4", help="pp ranges")
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(42)

    args = parse_args()
    world_size = args.world_size
    init_method = f"tcp://{args.ip}:{args.rpc_port}"
    model_path = args.model_path
    pp_ranges = args.pp_ranges

    mp.spawn(run_client, args=(world_size, model_path, pp_ranges, init_method), nprocs=world_size, join=True)
