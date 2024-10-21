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
    return torch.tensor(lst)


def build_mask(seq_len_list: List[Tuple[int, int]]) -> torch.Tensor:
    """
    构造多个请求的 casual mask
    @param seq_len_list: 每个请求的 seq_len

    @return: 一个 mask，形状为 total_length x total_length，其中 total_length 是所有请求的 seq_len 之和
    """
    mask_list = [torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) for (L, S) in seq_len_list]
    total_L, total_S = 0, 0
    for L, S in seq_len_list:
        total_L += L
        total_S += S

    combined_mask = torch.zeros((total_L, total_S), dtype=torch.bool)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.size(0), r_index : r_index + mask.size(1)] = mask
        l_index += mask.size(0)
        r_index += mask.size(1)

    return combined_mask
