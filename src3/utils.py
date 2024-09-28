import argparse
from typing import *

import torch
from transformers import AutoTokenizer


def setup_seed(seed):
    torch.manual_seed(seed)


def parse_range_string(s):
    try:
        ranges = s.split(",")
        result = []
        for r in ranges:
            start, end = map(int, r.split("-"))
            result.append((start, end))
        return result
    except:
        raise argparse.ArgumentTypeError("参数必须是形如 '1-2,3-4' 的范围字符串")


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


# 用于 RPC 请求
def call_remote_init(model_rref, start_layer_idx: int, end_layer_idx: int, model_path: str) -> torch.futures.Future:
    return model_rref.rpc_async().init_model(start_layer_idx, end_layer_idx, model_path)


def call_remote_forward(model_rref, hidden_states, uuid_str) -> torch.futures.Future:
    return model_rref.rpc_async().forward(hidden_states, uuid_str)
