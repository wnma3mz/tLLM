from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


def build_qkv(bs, num_heads, seq_len, head_dim):
    query = torch.randn(bs, num_heads, seq_len, head_dim)
    key = torch.randn(bs, num_heads, seq_len, head_dim)
    value = torch.randn(bs, num_heads, seq_len, head_dim)
    return query, key, value


if __name__ == "__main__":
    seq_len1, seq_len2 = 3, 4
    head_dim = 4
    query1, key1, value1 = build_qkv(1, 2, seq_len1, head_dim)
    query2, key2, value2 = build_qkv(1, 2, seq_len2, head_dim)
    position_ids1 = torch.arange(seq_len1).unsqueeze(0)
    position_ids2 = torch.arange(seq_len2).unsqueeze(0)

    rotary_emb = LlamaRotaryEmbedding(
        head_dim,
        max_position_embeddings=4096,
        base=10000,
    )
    cos1, sin1 = rotary_emb(value1, position_ids1)
    cos2, sin2 = rotary_emb(value2, position_ids2)

    query1_out, key1_out = apply_rotary_pos_emb(query1, key1, cos1, sin1, position_ids1)
    query2_out, key2_out = apply_rotary_pos_emb(query2, key2, cos2, sin2, position_ids2)

    # merge
    query = torch.cat([query1, query2], dim=-2)
    key = torch.cat([key1, key2], dim=-2)
    position_ids = torch.cat([position_ids1, position_ids2], dim=-1)
    cos, sin = rotary_emb(value1, position_ids)
    query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
    print("test merge rope for query", torch.allclose(query_out, torch.cat([query1_out, query2_out], dim=-2)))
    print("test merge rope for key", torch.allclose(key_out, torch.cat([key1_out, key2_out], dim=-2)))
