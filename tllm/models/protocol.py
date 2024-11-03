import asyncio
from dataclasses import dataclass, field
from typing import *

import torch

from tllm.generate.sampler_utils import SamplerUtils
from tllm.generate.sampling_params import SamplingParams

finish_reason_type = Literal["length", "stop", None]


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
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    comm_cost_time_list: Optional[List[float]] = None
    calc_cost_time_list: Optional[List[float]] = None


@dataclass
class SequenceRequestData:
    # 每个请求在输入输出模型的数据
    request_id: str
    input_ids: List[int]
    sampling_params: SamplingParams
    sampler: SamplerUtils

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
