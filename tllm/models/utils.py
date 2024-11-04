import json
import os
from typing import Dict, List, Set, Tuple, Union

from safetensors import safe_open
import torch

from tllm.models.protocol import GenerateEnd


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


def build_mask(seq_len_list: List[Tuple[int, int]]) -> torch.Tensor:
    """
    构造多个请求的 casual mask
    @param se   q_len_list: 每个请求的 seq_len

    @return: 一个 mask，形状为 total_length x total_length，其中 total_length 是所有请求的 seq_len 之和
    """
    mask_list = [
        torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if L > 1 else torch.ones((L, S), dtype=torch.bool)
        for (L, S) in seq_len_list
    ]
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


def read_from_safetensors(file_path: str, key_list: List[str]) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in key_list:
            if key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def load_master_weight(model_path: str) -> Dict[str, Union[torch.Tensor, "mx.array"]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    file_set = set()
    key_list = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        for key in key_list:
            if key in index["weight_map"]:
                file_set.add(index["weight_map"][key])
    else:
        file_set.add("model.safetensors")

    weight_dict = {}
    for file in file_set:
        weight_path = os.path.join(model_path, file)
        weight_dict.update(read_from_safetensors(weight_path, key_list))
    return weight_dict
