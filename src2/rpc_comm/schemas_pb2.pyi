from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Array(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, elements: _Optional[_Iterable[float]] = ...) -> None: ...

class Matrix(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Array]
    def __init__(
        self, rows: _Optional[_Iterable[_Union[Array, _Mapping]]] = ...
    ) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[Matrix]
    def __init__(
        self, layers: _Optional[_Iterable[_Union[Matrix, _Mapping]]] = ...
    ) -> None: ...

class BlockTensor(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[Tensor]
    def __init__(
        self, blocks: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ...
    ) -> None: ...

class MultiDimensionalArray(_message.Message):
    __slots__ = ("array", "matrix", "tensor", "block_tensor")
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    MATRIX_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    BLOCK_TENSOR_FIELD_NUMBER: _ClassVar[int]
    array: Array
    matrix: Matrix
    tensor: Tensor
    block_tensor: BlockTensor
    def __init__(
        self,
        array: _Optional[_Union[Array, _Mapping]] = ...,
        matrix: _Optional[_Union[Matrix, _Mapping]] = ...,
        tensor: _Optional[_Union[Tensor, _Mapping]] = ...,
        block_tensor: _Optional[_Union[BlockTensor, _Mapping]] = ...,
    ) -> None: ...

class LayerConfig(_message.Message):
    __slots__ = (
        "config",
        "layer_idx_start",
        "layer_idx_end",
        "tp_url_list",
        "tp_size",
        "layer_state_dict_dir",
    )
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_START_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_END_FIELD_NUMBER: _ClassVar[int]
    TP_URL_LIST_FIELD_NUMBER: _ClassVar[int]
    TP_SIZE_FIELD_NUMBER: _ClassVar[int]
    LAYER_STATE_DICT_DIR_FIELD_NUMBER: _ClassVar[int]
    config: _struct_pb2.Struct
    layer_idx_start: int
    layer_idx_end: int
    tp_url_list: _containers.RepeatedScalarFieldContainer[str]
    tp_size: int
    layer_state_dict_dir: str
    def __init__(
        self,
        config: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...,
        layer_idx_start: _Optional[int] = ...,
        layer_idx_end: _Optional[int] = ...,
        tp_url_list: _Optional[_Iterable[str]] = ...,
        tp_size: _Optional[int] = ...,
        layer_state_dict_dir: _Optional[str] = ...,
    ) -> None: ...

class ForwardData(_message.Message):
    __slots__ = ("uuid", "hidden_states")
    UUID_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    hidden_states: MultiDimensionalArray
    def __init__(
        self,
        uuid: _Optional[str] = ...,
        hidden_states: _Optional[_Union[MultiDimensionalArray, _Mapping]] = ...,
    ) -> None: ...

class MLPConfig(_message.Message):
    __slots__ = (
        "input_size",
        "output_size",
        "mlp_bias",
        "proj_name",
        "layer_idx",
        "tp_idx",
        "tp_size",
        "state_dict_path",
        "weight_data",
        "bias_data",
        "name",
    )
    INPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
    MLP_BIAS_FIELD_NUMBER: _ClassVar[int]
    PROJ_NAME_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    TP_IDX_FIELD_NUMBER: _ClassVar[int]
    TP_SIZE_FIELD_NUMBER: _ClassVar[int]
    STATE_DICT_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DATA_FIELD_NUMBER: _ClassVar[int]
    BIAS_DATA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    input_size: int
    output_size: int
    mlp_bias: bool
    proj_name: str
    layer_idx: int
    tp_idx: int
    tp_size: int
    state_dict_path: str
    weight_data: _any_pb2.Any
    bias_data: _any_pb2.Any
    name: str
    def __init__(
        self,
        input_size: _Optional[int] = ...,
        output_size: _Optional[int] = ...,
        mlp_bias: bool = ...,
        proj_name: _Optional[str] = ...,
        layer_idx: _Optional[int] = ...,
        tp_idx: _Optional[int] = ...,
        tp_size: _Optional[int] = ...,
        state_dict_path: _Optional[str] = ...,
        weight_data: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...,
        bias_data: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...,
        name: _Optional[str] = ...,
    ) -> None: ...

class MLPForwardData(_message.Message):
    __slots__ = (
        "proj_name",
        "tp_idx",
        "layer_idx",
        "hidden_states",
        "name",
        "cost_time",
    )
    PROJ_NAME_FIELD_NUMBER: _ClassVar[int]
    TP_IDX_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    COST_TIME_FIELD_NUMBER: _ClassVar[int]
    proj_name: str
    tp_idx: int
    layer_idx: int
    hidden_states: MultiDimensionalArray
    name: str
    cost_time: float
    def __init__(
        self,
        proj_name: _Optional[str] = ...,
        tp_idx: _Optional[int] = ...,
        layer_idx: _Optional[int] = ...,
        hidden_states: _Optional[_Union[MultiDimensionalArray, _Mapping]] = ...,
        name: _Optional[str] = ...,
        cost_time: _Optional[float] = ...,
    ) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(
        self, msg: _Optional[str] = ..., status: _Optional[int] = ...
    ) -> None: ...

class ForwardResponse(_message.Message):
    __slots__ = ("msg", "status", "output", "cost_time")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    COST_TIME_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    output: MultiDimensionalArray
    cost_time: float
    def __init__(
        self,
        msg: _Optional[str] = ...,
        status: _Optional[int] = ...,
        output: _Optional[_Union[MultiDimensionalArray, _Mapping]] = ...,
        cost_time: _Optional[float] = ...,
    ) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(
        self, msg: _Optional[str] = ..., status: _Optional[int] = ...
    ) -> None: ...

class MLPKeysResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(
        self, msg: _Optional[str] = ..., status: _Optional[int] = ...
    ) -> None: ...

class InitModelFlagResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: bool
    status: int
    def __init__(self, msg: bool = ..., status: _Optional[int] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
