import time
from typing import *

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from src3.commons.communicator import Communicator
from src3.model import MyLlamaModel
from src3.worker import ModelManager

# 使用 torch.dist 实现 张量并行，使用 torch.dist.rpc 实现 管道并行，通信时仅通信输入

ip = "localhost"
port = 29605

init_method = f"tcp://{ip}:{port}"
dist_init_method = f"tcp://{ip}:{port+2}"


def setup_seed(seed):
    torch.manual_seed(seed)


class MyLlamaForCausalLM(nn.Module):
    # 主控制程序
    def __init__(self, config):
        super().__init__()
        self.model = MyLlamaModel(config)
        self.vocab_size = config.vocab_size

        self.pp_size = 2

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, model_path: str, comm, model_rref_list, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # TODO: 暂定 PP=2
        config.decoder_start_layer_idx = 0
        config.decoder_end_layer_idx = config.num_hidden_layers // 2
        config.comm = comm
        cls.comm = comm
        cls.model_rref_list = model_rref_list
        config.offset = 0

        model = cls(config)
        from transformers import LlamaForCausalLM

        s1 = time.time()
        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()

        if comm.is_rank0():
            model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
            model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
            model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.model.load_state_dict(state_dict)
        print(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
        model.eval()
        return model

    def forward(self, inputs_embeds: torch.Tensor, position_ids, past_key_values):
        output = self.model(inputs_embeds, position_ids, past_key_values)
        hidden_states = output.last_hidden_state

        if self.comm.is_rank0():
            fut_list = []
            for i, model_rref in enumerate(self.model_rref_list[::-1]):
                if i == len(self.model_rref_list) - 1:
                    need_wait = True
                else:
                    need_wait = False
                fut = call_remote_forward(model_rref, hidden_states, "123", need_wait)
                fut_list.append(fut)
            # hidden_states = fut_list[0].wait()
            hidden_states = fut_list[-1]

        # 如果是 master 节点，必须一直等最后一个 PP 发送参数过来
        if self.comm.is_rank0():
            hidden_states = torch.tensor(hidden_states).to(inputs_embeds.dtype).to(self.norm.weight.device)
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits, output.past_key_values
        return None, output.past_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[List[int], Optional[torch.Tensor]]:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        bs, seq_len = input_ids.size()  # bs == 1
        past_key_values = None
        position_ids = None
        input_embeds = self.embed_tokens(input_ids)
        token_list: List[int] = []
        cnt = 0
        while True:
            self.comm.broadcast(input_embeds)
            # print(f"Rank: {dist.get_rank()}, input_embeds: {input_embeds}")
            if past_key_values is None:
                past_key_values = DynamicCache()
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            else:
                past_key_values_length = past_key_values.get_seq_length()
                position_ids = torch.arange(past_key_values_length, 1 + past_key_values_length, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0)

            logits, past_key_values = self(input_embeds, position_ids, past_key_values)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            if self.comm.is_rank0():
                next_token = torch.argmax(logits[:, -1], dim=1)
                token_list.append(next_token[0].tolist())
                print("token", token_list[-1])
                input_embeds = self.embed_tokens(next_token).unsqueeze(0)
            else:
                # 必须要加上这一行
                input_embeds = self.embed_tokens(torch.tensor([0])).unsqueeze(0)

        return token_list, None


def load_model_and_tokenizer(model_path: str, comm, model_rref_list) -> Tuple[MyLlamaForCausalLM, AutoTokenizer]:
    model = MyLlamaForCausalLM.from_pretrained(
        model_path, comm, model_rref_list, trust_remote_code=True, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


formatted_prompt = "### Human: {}### Assistant:"


def tokenize_message(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> List[int]:
    inputs = formatted_prompt.format(messages[0]["content"])
    # inputs = "Hello, how are you?"
    # inputs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(inputs, add_special_tokens=True)
    while input_ids[0] == input_ids[1] == tok.bos_token_id:
        # input_ids = input_ids[1:]
        input_ids.pop(0)
    return input_ids


def call_remote_init(model_rref, need_wait: bool = False):
    if need_wait:
        return model_rref.rpc_async().init_model().wait()
    else:
        return model_rref.rpc_async().init_model()


def call_remote_forward(model_rref, hidden_states, uuid_str, need_wait: bool = False):
    if need_wait:
        return model_rref.rpc_async().forward(hidden_states, uuid_str).wait()
    else:
        return model_rref.rpc_async().forward(hidden_states, uuid_str)


def run_client(rank, world_size, start_rank, end_rank):
    if rank not in range(start_rank, end_rank + 1):
        return

    options = rpc.TensorPipeRpcBackendOptions()
    options.init_method = init_method
    options.rpc_timeout = 180000
    # PP0（master）: decoder layer 第一部分 + embedding + lm head
    # PP: decoder layer 其他部分
    setup_seed(42)
    tp_world_size = end_rank - start_rank + 1
    comm = Communicator(dist_init_method, rank - start_rank, tp_world_size)
    rpc.init_rpc(name=f"client{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)

    # remote_rank = 0
    # model_rref = rpc.remote(f"worker{remote_rank}", ModelManager)
    # call_remote_init(model_rref, True)
    # model_rref_list = [model_rref]

    model_rref_list = []
    print("comm.rank", comm.rank)
    if comm.rank == 0:
        worker_start_rank, worker_end_rank = 0, 1
        for remote_rank in range(worker_start_rank, worker_end_rank + 1):
            model_rref = rpc.remote(f"worker{remote_rank}", ModelManager)
            model_rref_list.append(model_rref)
        # init and load model
        for i, model_rref in enumerate(model_rref_list):
            if i == len(model_rref_list) - 1:
                need_wait = True
            else:
                need_wait = False
            call_remote_init(model_rref, need_wait)

    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    s1 = time.time()
    model, tok = load_model_and_tokenizer(model_path, comm, model_rref_list)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    comm.print_rank0(f"input_ids: {input_ids}")

    s1 = time.time()
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    comm.print_rank0(f"token: {output[0]}")
    comm.print_rank0(f"cost time: {time.time() - s1}")

    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        comm.print_rank0(f"Time taken: {time.time() - s1}")
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    world_size = 4
    start_rank, end_rank = 2, 3
    mp.spawn(
        run_client,
        args=(
            world_size,
            start_rank,
            end_rank,
        ),
        nprocs=world_size,
        join=True,
    )
