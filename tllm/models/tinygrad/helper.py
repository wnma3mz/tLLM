from typing import List

from tinygrad.tensor import Tensor


def greedy_decode(logits: Tensor) -> List[int]:
    return logits.argmax(axis=-1).tolist()
