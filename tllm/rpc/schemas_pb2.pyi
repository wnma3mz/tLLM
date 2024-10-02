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
    def __init__(self, rows: _Optional[_Iterable[_Union[Array, _Mapping]]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[Matrix]
    def __init__(self, layers: _Optional[_Iterable[_Union[Matrix, _Mapping]]] = ...) -> None: ...

class BlockTensor(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[Tensor]
    def __init__(self, blocks: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ...) -> None: ...

class BFloat16Tensor(_message.Message):
    __slots__ = ("data", "shape")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

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

class ModelConfig(_message.Message):
    __slots__ = ("model_name", "pp_rank", "layer_idx_start", "layer_idx_end", "master_url", "next_pp_rank")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PP_RANK_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_START_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_END_FIELD_NUMBER: _ClassVar[int]
    MASTER_URL_FIELD_NUMBER: _ClassVar[int]
    NEXT_PP_RANK_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    pp_rank: int
    layer_idx_start: int
    layer_idx_end: int
    master_url: str
    next_pp_rank: int
    def __init__(
        self,
        model_name: _Optional[str] = ...,
        pp_rank: _Optional[int] = ...,
        layer_idx_start: _Optional[int] = ...,
        layer_idx_end: _Optional[int] = ...,
        master_url: _Optional[str] = ...,
        next_pp_rank: _Optional[int] = ...,
    ) -> None: ...

class ForwardRequest(_message.Message):
    __slots__ = ("uuid", "hidden_states")
    UUID_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    hidden_states: BFloat16Tensor
    def __init__(
        self, uuid: _Optional[str] = ..., hidden_states: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...
    ) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class ForwardResponse(_message.Message):
    __slots__ = ("msg", "status", "output", "cost_time")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    COST_TIME_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    output: BFloat16Tensor
    cost_time: float
    def __init__(
        self,
        msg: _Optional[str] = ...,
        status: _Optional[int] = ...,
        output: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
        cost_time: _Optional[float] = ...,
    ) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

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
