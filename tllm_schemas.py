from typing import *

import torch
from pydantic import BaseModel
from transformers.cache_utils import Cache


class LayerConfig(BaseModel):
    config: dict
    layer_idx_start: int
    layer_idx_end: int
    tp_url_list: List[str]
    tp_size: int
    state_dict_path: str
    layer_state_dict_path: Optional[str] = None


class ForwardData(BaseModel):
    uuid: str
    hidden_states: List


class MLPForwardData(BaseModel):
    proj_name: str
    tp_idx: int
    layer_idx: int
    hidden_states: Optional[List] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.name = f"{self.proj_name}_{self.layer_idx}_{self.tp_idx}"


class MLPConfig(BaseModel):
    input_size: int
    output_size: int
    mlp_bias: bool
    proj_name: str
    layer_idx: int
    tp_idx: int
    tp_size: int
    state_dict_path: Optional[str] = None
    weight_data: Optional[List] = None
    bias_data: Optional[List] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.name = f"{self.proj_name}_{self.layer_idx}_{self.tp_idx}"
