from typing import List

import torch


def greedy_decode(logits: torch.Tensor) -> List[int]:
    # logits shape: [seq_len, vocab_size]
    return torch.argmax(logits, dim=-1).tolist()
