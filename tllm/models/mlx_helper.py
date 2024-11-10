from typing import List

import mlx.core as mx
import numpy as np


def greedy_decode(logits: mx.array) -> List[int]:
    # logits shape: [seq_len, vocab_size]
    x = mx.argmax(logits, axis=-1)
    return x.tolist()  # TODO: first requests is too slow
