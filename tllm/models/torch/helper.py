import itertools
from typing import Dict, List

from safetensors.torch import load_file as safe_load
import torch

from tllm import DEVICE, DTYPE
from tllm.commons.attn import ATTN_TYPE
from tllm.commons.cache import AttentionData
from tllm.commons.cache_manager import CacheManager
from tllm.schemas import SeqInput


def greedy_decode(logits: "torch.Tensor") -> List[int]:
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


def read_from_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    return safe_load(file_path, device="cpu")


if ATTN_TYPE == "xformers":
    from xformers.ops import fmha


class TorchCacheManager(CacheManager):
    def build_forward_cache(self, hidden_states: torch.Tensor, seq_input: SeqInput) -> torch.Tensor:
        # request_cache = RequestsCache(num_layers, max_seq_len, num_key_value_heads, head_dim)
        q_len_list, k_len_list, position_ids_list, _ = self.request_cache.build(seq_input, self.cache)

        self.hit_cache_flag = None

        if ATTN_TYPE == "flash_attention":
            attn_mask = {
                "cu_seqlens_q": torch.tensor([0] + list(itertools.accumulate(q_len_list)), dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0] + list(itertools.accumulate(k_len_list)), dtype=torch.int32),
                "max_seqlen_q": max(q_len_list),
                "max_seqlen_k": max(k_len_list),
            }
        # elif ATTN_TYPE == "xformers":
        #     attn_mask = fmha.BlockDiagonalMask.from_seqlens(q_seqlen=q_len_list, kv_seqlen=k_len_list)
        else:
            attn_mask = build_mask(q_len_list, k_len_list)

        self.q_len_list = q_len_list
        self.hit_cache_len_list = None

        if ATTN_TYPE == "flash_attention":
            attn_mask = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in attn_mask.items()}
        else:
            attn_mask = attn_mask.to(DEVICE)

        self.attn_data = AttentionData(
            request_cache=self.request_cache,
            attn_mask=attn_mask,
            uuid_list=seq_input.uuid_list,
            position_ids=torch.cat(position_ids_list, dim=-1),
        )

        hidden_states = hidden_states.to(DTYPE).to(DEVICE)
        return hidden_states

    def get_last_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        split_len_list = self.q_len_list
        # if self.hit_cache_flag:
        #     q_start = 0
        #     for i, (q_len, hit_cache_len) in enumerate(zip(self.q_len_list, self.hit_cache_len_list)):
        #         if hit_cache_len != -1:
        #             split_len_list[i] = q_len - hit_cache_len
        #         q_start += q_len
        if self.is_end_pp:
            # 只取最后一个 token 的 hidden_states
            seq_hidden_states = torch.split(hidden_states, [seq_len for seq_len in split_len_list], dim=0)
            hidden_states = torch.cat([x[-1:, :] for x in seq_hidden_states], dim=0)
        return hidden_states
