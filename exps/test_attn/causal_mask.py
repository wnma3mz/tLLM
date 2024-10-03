from typing import List

import torch


def merge_mask(mask_list: List[torch.Tensor], total_length: int) -> torch.Tensor:
    combined_mask = torch.zeros((total_length, total_length), dtype=torch.bool)

    start_index = 0
    for mask in mask_list:
        combined_mask[start_index : start_index + mask.size(0), start_index : start_index + mask.size(1)] = mask
        start_index += mask.size(0)

    combined_attn_bias = torch.zeros(total_length, total_length, dtype=torch.float)
    combined_attn_bias.masked_fill_(combined_mask.logical_not(), float("-inf"))
    return combined_attn_bias


def build_mask(mask: torch.Tensor) -> torch.Tensor:
    attn_bias = torch.zeros(mask.size(0), mask.size(1), dtype=torch.float)
    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
    return attn_bias


def build_qkv(bs, num_heads, seq_len, head_dim):
    query = torch.randn(bs, num_heads, seq_len, head_dim)
    key = torch.randn(bs, num_heads, seq_len, head_dim)
    value = torch.randn(bs, num_heads, seq_len, head_dim)
    return query, key, value


if __name__ == "__main__":
    seq_len1, seq_len2 = 3, 4
    temp_mask = torch.ones(seq_len1, seq_len1, dtype=torch.bool).tril(diagonal=0)
    temp_mask2 = torch.ones(seq_len2, seq_len2, dtype=torch.bool).tril(diagonal=0)

    combined_attn_bias = merge_mask([temp_mask, temp_mask2], seq_len1 + seq_len2)

    # bs, num_heads, seq_len, head_dim
    query1, key1, value1 = build_qkv(1, 2, seq_len1, 4)
    base_out1 = torch.nn.functional.scaled_dot_product_attention(query1, key1, value1, is_causal=True)

    query2, key2, value2 = build_qkv(1, 2, seq_len2, 4)
    base_out2 = torch.nn.functional.scaled_dot_product_attention(query2, key2, value2, is_causal=True)

    query = torch.cat([query1, query2], dim=-2)
    key = torch.cat([key1, key2], dim=-2)
    value = torch.cat([value1, value2], dim=-2)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=combined_attn_bias)
    out1, out2 = torch.split(out, [seq_len1, seq_len2], dim=-2)

    print("torch.allclose(base_out1, out1)", torch.allclose(base_out1, out1))
    print("torch.allclose(base_out2, out2)", torch.allclose(base_out2, out2))
