import os
import time
from typing import *

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.protocol import SeqInput
from tllm.models.torch.llama import MyLlamaModel

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


def build_input(uuid: str, seq_len: int) -> Tuple[SeqInput, torch.Tensor]:
    uuid_list = [uuid]
    seq_len_list = [seq_len]
    hidden_states = torch.randn(1, seq_len, 2048).to(dtype)
    return SeqInput(uuid_list=uuid_list, seq_len_list=seq_len_list), hidden_states


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
    cat_output = model(cat_hidden_states, SeqInput(uuid_list=["11", "22"], seq_len_list=[3, 2]))

    output_12 = torch.cat([output_1, output_2], dim=1)
    # 请求全处于 prefill 阶段
    print(f"is_same: {torch.allclose(output_12, cat_output)}")

    # 请求部分处于 prefill 阶段，部分处于 decode 阶段
    cat_hidden_states = torch.cat([hidden_states_1, hidden_states_22], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_list=["33", "22"], seq_len_list=[3, 1]))
    output_122 = torch.cat([output_1, output_22], dim=1)
    print(f"is_same: {torch.allclose(output_122, cat_output)}")

    # 请求部分处于 prefill 阶段，部分处于 decode 阶段
    cat_hidden_states = torch.cat([hidden_states_11, hidden_states_2], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_list=["11", "44"], seq_len_list=[1, 2]))
    output_122 = torch.cat([output_11, output_2], dim=1)
    print(f"is_same: {torch.allclose(output_122, cat_output)}")

    # 请求全处于 decode 阶段
    cat_hidden_states = torch.cat([hidden_states_11, hidden_states_22], dim=1)
    cat_output = model(cat_hidden_states, SeqInput(uuid_list=["11", "22"], seq_len_list=[1, 1]))
    output_1122 = torch.cat([output_11, output_22], dim=1)
    print(output_1122)
    print(cat_output)
    print("diff ", torch.sum(output_1122 - cat_output))
    print(f"is_same: {torch.allclose(output_1122, cat_output)}")  # 存在误差，会返回 False。TODO
