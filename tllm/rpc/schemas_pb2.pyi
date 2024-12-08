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

class StatusRequest(_message.Message):
    __slots__ = ("uuid", "seq_len", "pp_idx", "cost_time")
    UUID_FIELD_NUMBER: _ClassVar[int]
    SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    PP_IDX_FIELD_NUMBER: _ClassVar[int]
    COST_TIME_FIELD_NUMBER: _ClassVar[int]
    uuid: _containers.RepeatedScalarFieldContainer[str]
    seq_len: _containers.RepeatedScalarFieldContainer[int]
    pp_idx: int
    cost_time: float
    def __init__(
        self,
        uuid: _Optional[_Iterable[str]] = ...,
        seq_len: _Optional[_Iterable[int]] = ...,
        pp_idx: _Optional[int] = ...,
        cost_time: _Optional[float] = ...,
    ) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class ForwardResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetConfigRequest(_message.Message):
    __slots__ = ("forward_url", "master_url", "pp_rank")
    FORWARD_URL_FIELD_NUMBER: _ClassVar[int]
    MASTER_URL_FIELD_NUMBER: _ClassVar[int]
    PP_RANK_FIELD_NUMBER: _ClassVar[int]
    forward_url: str
    master_url: str
    pp_rank: int
    def __init__(
        self, forward_url: _Optional[str] = ..., master_url: _Optional[str] = ..., pp_rank: _Optional[int] = ...
    ) -> None: ...

class SetConfigResponse(_message.Message):
    __slots__ = ("msg", "status")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    def __init__(self, msg: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class ImageForwardRequest(_message.Message):
    __slots__ = ("uuid", "hidden_states", "encoder_hidden_states", "text_embeddings", "image_rotary_emb")
    UUID_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    ENCODER_HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    TEXT_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_ROTARY_EMB_FIELD_NUMBER: _ClassVar[int]
    uuid: _containers.RepeatedScalarFieldContainer[str]
    hidden_states: BFloat16Tensor
    encoder_hidden_states: BFloat16Tensor
    text_embeddings: BFloat16Tensor
    image_rotary_emb: BFloat16Tensor
    def __init__(
        self,
        uuid: _Optional[_Iterable[str]] = ...,
        hidden_states: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
        encoder_hidden_states: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
        text_embeddings: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
        image_rotary_emb: _Optional[_Union[BFloat16Tensor, _Mapping]] = ...,
    ) -> None: ...

class ImageForwardResponse(_message.Message):
    __slots__ = ("msg", "status", "image")
    MSG_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    msg: str
    status: int
    image: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self, msg: _Optional[str] = ..., status: _Optional[int] = ..., image: _Optional[_Iterable[str]] = ...
    ) -> None: ...
