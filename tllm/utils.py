import argparse
import socket
from typing import *

import torch

from tllm.schemas import NodeConfig


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


# 用于 RPC 请求
def call_remote_init(model_rref, node_config: NodeConfig) -> torch.futures.Future:
    return model_rref.rpc_async().init_model(node_config)


def call_remote_forward(
    model_rref, hidden_states: Optional[torch.Tensor], shape_hidden_states: Tuple[int], uuid_str: str
) -> torch.futures.Future:
    return model_rref.rpc_async().forward(hidden_states, shape_hidden_states, uuid_str)


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def create_decoder_attention_mask(size: int) -> torch.Tensor:
    # Create a lower triangular matrix with ones below the diagonal
    mask = torch.tril(torch.ones(size, size)).transpose(0, 1)
    # Fill the diagonal with ones as well
    mask = mask.masked_fill(mask == 0, float("-inf"))
    return mask


def tensor_to_list(tensor: Optional[torch.Tensor]) -> List:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.float().cpu().detach().numpy().tolist()


def list_to_tensor(lst: Optional[List]) -> torch.Tensor:
    if lst is None:
        return None
    if not isinstance(lst, list):
        return lst
    return torch.tensor(lst)
