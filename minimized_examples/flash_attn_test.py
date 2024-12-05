import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
from xformers.components.attention.core import scaled_dot_product_attention

from tllm.models.torch.helper import build_mask


def prefill_test():
    seq_len, n_heads, head_dim = 20, 32, 256
    q = torch.rand((seq_len, n_heads, head_dim), device=device, dtype=dtype)
    k = torch.rand((seq_len, n_heads, head_dim), device=device, dtype=dtype)
    v = torch.rand((seq_len, n_heads, head_dim), device=device, dtype=dtype)

    attn_mask = build_mask([(seq_len, seq_len)]).to(device)

    base_out = scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), att_mask=attn_mask)
    q_seq_start_loc, k_seq_start_loc = torch.tensor([0, seq_len], device=device, dtype=torch.int32), torch.tensor(
        [0, seq_len], device=device, dtype=torch.int32
    )
    q_seq_len, k_seq_len = seq_len, seq_len
    flash_out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=q_seq_start_loc,
        cu_seqlens_k=k_seq_start_loc,
        max_seqlen_q=q_seq_len,
        max_seqlen_k=k_seq_len,
        causal=True,
    )

    print(base_out.shape, base_out[0])

    print(flash_out.shape, flash_out.transpose(0, 1)[0])
    print("diff sum", torch.sum(torch.abs(base_out[0] - flash_out.transpose(0, 1)[0])))


def prefilling_decode_test():
    # seq1 为 prefill， seq2 为 decode

    n_heads, head_dim = 32, 256

    seq_len1, seq_len2, cache_seq_len2 = 10, 1, 5
    total_q, total_k = seq_len1 + seq_len2, seq_len1 + cache_seq_len2
    q = torch.rand((total_q, n_heads, head_dim), device=device, dtype=dtype)
    k = torch.rand((total_k, n_heads, head_dim), device=device, dtype=dtype)
    v = torch.rand((total_k, n_heads, head_dim), device=device, dtype=dtype)

    attn_mask = build_mask([(seq_len1, seq_len1), (seq_len2, cache_seq_len2)]).to(device)

    base_out = scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), att_mask=attn_mask)
    q_seq_start_loc, k_seq_start_loc = torch.tensor(
        [0, seq_len1, total_q], device=device, dtype=torch.int32
    ), torch.tensor([0, seq_len1, total_k], device=device, dtype=torch.int32)
    q_seq_len, k_seq_len = max(seq_len1, seq_len2), max(seq_len1, cache_seq_len2)
    flash_out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=q_seq_start_loc,
        cu_seqlens_k=k_seq_start_loc,
        max_seqlen_q=q_seq_len,
        max_seqlen_k=k_seq_len,
        causal=True,
    )

    print(base_out.shape, base_out[0])

    print(flash_out.shape, flash_out.transpose(0, 1)[0])
    print("diff sum", torch.sum(torch.abs(base_out[0] - flash_out.transpose(0, 1)[0])))


if __name__ == "__main__":
    # pass
    device = "cuda:0"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    prefill_test()
    # prefilling_decode_test()
