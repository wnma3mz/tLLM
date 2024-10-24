import os
import time
from typing import *

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.generate.decode_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.llama import MyLlamaModel
from tllm.rpc.protocol import SeqInput

# PYTHONPATH='./' python3 minimized_examples/continue_batch/main.py


def func(messages: List[Dict[str, str]]):
    input_id_list = tok.preprocess(messages=messages).input_ids
    input_ids = torch.tensor(input_id_list).unsqueeze(0)

    print("input_ids: ", input_ids)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print("generate token: ", output[0])


def continue_batch(input_ids: List[int], mask: torch.Tensor):
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print("generate token: ", output[0])


def build_input(uuid_str: str, seq_len: int) -> Tuple[SeqInput, torch.Tensor]:
    uuid_str_list = [uuid_str]
    seq_len_list = [seq_len]
    hidden_states = torch.randn(1, seq_len, 2048).to(dtype)
    return SeqInput(uuid_str_list=uuid_str_list, seq_len_list=seq_len_list), hidden_states


if __name__ == "__main__":
    torch.manual_seed(0)
    model_path = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.decoder_start_layer_idx = 0
    config.decoder_end_layer_idx = 1
    config.comm = SingleNodeCommunicator()

    dtype = torch.bfloat16
    s1 = time.time()
    state_dict = LlamaForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=dtype, low_cpu_mem_usage=True
    ).state_dict()
    model = MyLlamaModel(config).to(dtype)
    model.load_state_dict(state_dict)

    decode_hidden_states = torch.randn(1, 1, 2048).to(dtype)

    seq_input, hidden_states_1 = build_input("123", 3)
    output_1 = model(hidden_states_1, seq_input)

    seq_input, hidden_states_2 = build_input("234", 2)
    output_2 = model(hidden_states_2, seq_input)

    seq_input, hidden_states_11 = build_input("123", 1)
    output_11 = model(hidden_states_11, seq_input)

    seq_input, hidden_states_22 = build_input("234", 1)
    output_22 = model(hidden_states_22, seq_input)

    print(f"output_11: {output_11}; output_22: {output_22}")

    # assert False
    cat_hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_str_list=["11", "22"], seq_len_list=[3, 2]))

    output_12 = torch.cat([output_1, output_2], dim=1)
    # 请求全处于 prefill 阶段
    print(f"is_same: {torch.allclose(output_12, cat_output)}")

    # 请求部分处于 prefill 阶段，部分处于 decode 阶段
    cat_hidden_states = torch.cat([hidden_states_1, hidden_states_22], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_str_list=["33", "22"], seq_len_list=[3, 1]))
    output_122 = torch.cat([output_1, output_22], dim=1)
    print(f"is_same: {torch.allclose(output_122, cat_output)}")

    # 请求全处于 decode 阶段
    cat_hidden_states = torch.cat([hidden_states_11, hidden_states_22], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_str_list=["11", "22"], seq_len_list=[1, 1]))
    output_1122 = torch.cat([output_11, output_22], dim=1)
    print(output_1122)
    print(cat_output)
    print(f"is_same: {torch.allclose(output_1122, cat_output)}")  # 存在误差，会返回 False。TODO

    assert False
    tok = TokenizerUtils(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
    )

    model.eval()

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    # input_ids:  tensor([[128000, 128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,
    #        2696,     25,   6790,    220,   2366,     18,    198,  15724,   2696,
    #          25,    220,   1691,   5020,    220,   2366,     19,    271, 128009,
    #      128006,    882, 128007,    271,   9906,     11,   1268,    527,    499,
    #          30, 128009, 128006,  78191, 128007,    271]])
    # 40,   2846,   1120,
    #        264,   4221,   1646,     11,    358,   1541,    956,    617,  16024,
    #        477,  21958,     11,    719,    358,   2846,  31301,  10489])
    func(messages)

    messages = [{"role": "user", "content": "What's your name?"}]
    # input_ids:  tensor([[128000, 128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,
    #        2696,     25,   6790,    220,   2366,     18,    198,  15724,   2696,
    #          25,    220,   1691,   5020,    220,   2366,     19,    271, 128009,
    #      128006,    882, 128007,    271,   3923,    596,    701,    836,     30,
    #      128009, 128006,  78191, 128007,    271]])
    # 40,   2846,    459,  21075,
    #      11478,   1646,   3967,    439,    445,  81101,     13,    445,  81101,
    #      13656,    369,    330,  35353,  11688,   5008,  16197])
    func(messages)
