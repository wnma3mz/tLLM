import itertools
from typing import Dict, List, Tuple

from safetensors import safe_open
import torch
import torch.nn as nn

from tllm.commons.attn import get_attention_implementation
from tllm.commons.cache import AttentionData, CacheManager, RequestsCache
from tllm.schemas import SeqInput


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


class EmptyLayer(nn.Module):
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data,
    ) -> torch.Tensor:
        return hidden_states


_, attention_type = get_attention_implementation()

if attention_type == "xformers":
    from xformers.ops import fmha


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    position_ids_list, q_len_list, k_len_list = [], [], []
    for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            # kv_cache 是整个历史的 kv_cache
            # 当 q_len 为 1 时，直接使用 kv_cache，使用历史的全部 token kv cache
            # TODO: 当 q_len > 1 时，表示只需要使用前 q_len 的 kv_cache，后面的 kv_cache 需要重新计算
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            position_ids = torch.tensor([cache_seq_len], dtype=torch.long)
            k_len_list.append(cache_seq_len + q_len)
        else:
            layer_cache_list = None
            position_ids = torch.arange(q_len, dtype=torch.long)
            k_len_list.append(q_len)
        q_len_list.append(q_len)
        request_cache.add(uuid, q_len, layer_cache_list)
        position_ids_list.append(position_ids)

    if attention_type == "flash_attention":
        attn_mask = {
            "cu_seqlens_q": torch.tensor([0] + list(itertools.accumulate(q_len_list)), dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0] + list(itertools.accumulate(k_len_list)), dtype=torch.int32),
            "max_seqlen_q": max(q_len_list),
            "max_seqlen_k": max(k_len_list),
        }
    # elif attention_type == "xformers":
    #     attn_mask = fmha.BlockDiagonalMask.from_seqlens(q_seqlen=q_len_list, kv_seqlen=k_len_list)
    else:
        attn_mask = build_mask(q_len_list, k_len_list)
    return AttentionData(
        request_cache=request_cache,
        attn_mask=attn_mask,
        uuid_list=seq_input.uuid_list,
        position_ids=torch.cat(position_ids_list, dim=-1),
    )
