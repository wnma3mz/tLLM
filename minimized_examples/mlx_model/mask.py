from mlx_lm.models.cache import KVCache

from tllm.models.mlx_llama import build_mlx_mask
from tllm.models.utils import build_mask

if __name__ == "__main__":
    mask = build_mask([(2, 3), (1, 4)])
    mlx_mask = build_mlx_mask([(2, 3), (1, 4)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)

    prefill_seq_len = 3
    mask = build_mask([(prefill_seq_len, prefill_seq_len)])
    mlx_mask = build_mlx_mask([(prefill_seq_len, prefill_seq_len)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)

    mask = build_mask([(1, 1 + prefill_seq_len)])
    mlx_mask = build_mlx_mask([(1, 1 + prefill_seq_len)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)
