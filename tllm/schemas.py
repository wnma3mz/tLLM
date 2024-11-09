from typing import Union

import numpy as np
from pydantic import BaseModel

MIX_TENSOR = Union[np.ndarray, "torch.Tensor", "mx.array"]


class NodeConfig(BaseModel):
    start_layer_idx: int
    end_layer_idx: int
    checkpoint_path: str
    prev_rank: int
    next_start_rank: int
    next_end_rank: int
    rank: int = None
