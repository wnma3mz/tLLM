from typing import *

from pydantic import BaseModel


class NodeConfig(BaseModel):
    start_layer_idx: int
    end_layer_idx: int
    checkpoint_path: str
    prev_rank: int
    next_start_rank: int
    next_end_rank: int
    rank: int = None


class LayerConfig(BaseModel):
    config: dict
    layer_idx_start: int
    layer_idx_end: int
    tp_url_list: List[str]
    tp_size: int
    layer_state_dict_dir: str


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
