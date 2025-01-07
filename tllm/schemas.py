import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

from PIL import Image
import numpy as np
from pydantic import BaseModel

finish_reason_type = Literal["length", "stop", None]
modal_type = Literal["text", "image_url"]

MIX_TENSOR = Union[np.ndarray, "torch.Tensor", "mx.array"]


class UrlItem(BaseModel):
    url: Optional[str] = None
    file_path: Optional[str] = None
    base64: Optional[str] = None


class MultiModalContent(BaseModel):
    type: modal_type
    text: Optional[str] = None
    image_url: Optional[UrlItem] = None


MESSAGES = List[Dict[str, Union[str, List[MultiModalContent]]]]


class SamplingParams:
    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        min_tokens: int = 0,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ):
        assert n == 1, "Only n=1 is supported"
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.n = n
        self.stop_token_ids = stop_token_ids


@dataclass
class SeqInput:
    uuid_list: List[str]
    seq_len_list: List[int]


@dataclass
class GenerateEnd:
    finish_reason: finish_reason_type
    is_end: bool


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: Tuple[int, ...]
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


@dataclass
class RequestOutput:
    # 转换为 HTTP 接口的数据结构
    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        finished: bool,
        prompt_logprobs: Optional[List[float]] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished


@dataclass
class ForwardResult:
    hidden_states: MIX_TENSOR
    comm_cost_time: float
    calc_cost_time: float


@dataclass
class ImageResponseOutput:
    outputs: List[str]


@dataclass
class ImageRequestData:
    request_id: str

    prompt: str
    config: "ImageGenerationConfig"
    seed: int
    generate_iter: int = 0
    input_embeds: Optional["EmbeddingResult"] = None
    runtime_config = None

    is_stop: bool = False
    finish_reason_list: Optional[List[str]] = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    timeout: int = 100000  # 请求的总超时时间
    start_time: Optional[float] = None

    output_base64: str = ""

    def to_request_output(self) -> RequestOutput:
        # TODO: streaming support
        return ImageResponseOutput([self.output_base64])


@dataclass
class SequenceRequestData:
    # 每个请求在输入输出模型的数据
    request_id: str
    input_ids: List[int]
    sampling_params: SamplingParams

    multi_modal_inputs: Optional[Dict[str, List[Image.Image]]] = None
    history_request_id: Optional[str] = None
    finish_reason_list: Optional[List[str]] = None

    output_ids: Optional[List[int]] = None  # 最终生成的 token id
    output_text: Optional[str] = None  # 最终生成的 text

    generate_text: Optional[str] = None  # 每次生成的 text

    ttft_cost_time: Optional[float] = None
    decode_start_ts: Optional[float] = None
    timeout: int = 100000  # 请求的总超时时间
    is_stop: bool = False
    is_prefill: bool = True
    q_len: int = -1

    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def __post_init__(self):
        self.output_ids = []
        self.output_text = ""
        self.generate_text = None
        self.finish_reason_list = [None] * self.sampling_params.n
        self.decode_start_ts = None
        self.q_len = len(self.input_ids) if self.q_len == -1 else self.q_len

    def __repr__(self) -> str:
        return f"request_id={self.request_id}; output_ids={self.output_ids}"

    def to_request_output(self) -> RequestOutput:
        if not self.is_stop:
            return RequestOutput(
                self.request_id,
                None,
                self.input_ids,
                [
                    CompletionOutput(
                        index=index,
                        text=self.generate_text,
                        token_ids=self.output_ids[-1],
                        finish_reason=self.finish_reason_list[index],
                    )
                    for index in range(self.sampling_params.n)
                ],
                finished=self.is_stop,
                prompt_logprobs=None,
            )
        return RequestOutput(
            request_id=self.request_id,
            prompt=None,
            prompt_token_ids=self.input_ids,
            outputs=[
                CompletionOutput(
                    index=index,
                    text=self.output_text,
                    token_ids=tuple(self.output_ids),
                    finish_reason=self.finish_reason_list[index],
                )
                for index in range(self.sampling_params.n)
            ],
            finished=True,
            prompt_logprobs=None,
        )


class RegisterClientResponse(BaseModel):
    pp_rank: int
    start_idx: int
    end_idx: int
    msg: Optional[str] = None
    repo_path: Optional[str] = None

    def __repr__(self):
        return f"pp_rank={self.pp_rank} layer={self.start_idx}-{self.end_idx} model={self.repo_path}"

    def __str__(self):
        return self.__repr__()


class RegisterClientRequest(BaseModel):
    client_id: str
    host: List[str]
    port: int
    pp_rank: Optional[int] = -1
    start_idx: Optional[int] = -1
    end_idx: Optional[int] = -1

    def __repr__(self):
        return f"ip={self.host} port={self.port} pp_rank={self.pp_rank} layer={self.start_idx}-{self.end_idx}"

    def __str__(self):
        return self.__repr__()


class InitModelRequest(BaseModel):
    client_id: str
    pp_rank: int
    start_idx: int
    end_idx: int

    def __repr__(self):
        return f"pp_rank={self.pp_rank} layer={self.start_idx}-{self.end_idx}"

    def __str__(self):
        return self.__repr__()


class InitModelResponse(BaseModel):
    msg: str
    status: int


class ClientData(BaseModel):
    client_id: str
    host: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    pp_rank: Optional[int] = None
