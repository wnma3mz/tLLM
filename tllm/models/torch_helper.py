from typing import Dict, List, Tuple

from safetensors import safe_open
import torch


def greedy_decode(logits: torch.Tensor) -> List[int]:
    # logits shape: [seq_len, vocab_size]
    return torch.argmax(logits, dim=-1).tolist()


def build_mask(q_len_list: List[int], k_len_list: List[int]) -> torch.Tensor:
    """
    构造多个请求的 casual mask
    @param
        q_len_list: 每个请求的 seq_len
        k_len_list: 每个请求的 seq_len

    @return: 一个 mask，形状为 total_length x total_length，其中 total_length 是所有请求的 seq_len 之和
    """
    mask_list = [
        torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if L > 1 else torch.ones((L, S), dtype=torch.bool)
        for (L, S) in zip(q_len_list, k_len_list)
    ]

    combined_mask = torch.zeros((sum(q_len_list), sum(k_len_list)), dtype=torch.bool)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.size(0), r_index : r_index + mask.size(1)] = mask
        l_index += mask.size(0)
        r_index += mask.size(1)

    return combined_mask


def read_from_safetensors(file_path: str, key_list: List[str]) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for prefix_key in key_list:
                if key.startswith(prefix_key):
                    tensors[key] = f.get_tensor(key)
    return tensors
