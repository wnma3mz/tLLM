from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

from google.protobuf import descriptor as _descriptor, message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class BFloat16Tensor(_message.Message):
    __slots__ = ("data", "shape")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

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
    __slots__ = ("uuid", "seq_len", "hidden_states")
    UUID_FIELD_NUMBER: _ClassVar[int]
    SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    uuid: _containers.RepeatedScalarFieldContainer[str]
    seq_len: _containers.RepeatedScalarFieldContainer[int]
    hidden_states: BFloat16Tensor
    def __init__(
        self,
        uuid: _Optional[_Iterable[str]] = ...,
        seq_len: _Optional[_Iterable[int]] = ...,
        hidden_states: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
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
