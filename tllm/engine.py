import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import *

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tllm.commons.convert import deserialize_bfloat16_tensor, serialize_bfloat16_tensor
from tllm.generate.decode_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.rpc.manager import RPCManager
from tllm.rpc.protocol import SeqInput

finish_reason_type = Literal["length", "stop", None]

logging.basicConfig(level=logging.INFO)


@dataclass
class GenerateResult:
    output_ids: List[int]
    finish_reason: Optional[finish_reason_type] = None
    output_text: Optional[str] = None
    ttft: Optional[float] = None


@dataclass
class GenerateEnd:
    finish_reason: finish_reason_type
    is_end: bool


@dataclass
class ChatCompletionResponse:
    token: List[int]
    cost_time: float
    finish_reason: Optional[str]
    usage: Dict[str, int]
    text: str
    ttft: float


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
    comm_cost_time_list: Optional[List[float]] = None
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class SequenceRequestData:
    # 每个请求在输入输出模型的数据
    request_id: str
    n: int = 1  # 需要生成的句子数量
    input_ids: Optional[List[int]] = None  # 输入的 token id
    finish_reason_list: Optional[List[str]] = None

    sampler: Optional[Callable] = None
    sampling_params: Optional[Dict] = None

    output_ids: Optional[List[int]] = None  # 最终生成的 token id
    output_text: Optional[str] = None  # 最终生成的 text

    generate_ids: Optional[List[int]] = None  # 每次生成的 token id
    generate_texts: Optional[List[str]] = None  # 每次生成的 text

    ttft_cost_time: Optional[List[float]] = None
    tpot_cost_time: Optional[List[float]] = None
    timeout: int = 100000  # 请求的总超时时间
    is_stop: bool = False

    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def __post_init__(self):
        self.sampling_params = dict()
        self.output_ids = []
        self.output_text = ""
        self.generate_ids = []
        self.generate_texts = []
        self.finish_reason_list = [None] * self.n

    def __repr__(self) -> str:
        return f"request_id={self.request_id}; output_ids={self.output_ids}"

    def to_request_output(self) -> RequestOutput:
        if not self.is_stop:
            return RequestOutput(
                self.request_id,
                None,
                self.input_ids.tolist(),
                [
                    CompletionOutput(
                        index=index,
                        text=self.generate_texts[index],
                        token_ids=self.generate_ids[index],
                        finish_reason=self.finish_reason_list[index],
                    )
                    for index in range(self.n)
                ],
                self.is_stop,
                None,
            )
        return RequestOutput(
            request_id=self.request_id,
            prompt=None,
            prompt_token_ids=self.input_ids.tolist(),
            outputs=[
                CompletionOutput(
                    index=index,
                    text=self.output_text,
                    token_ids=tuple(self.output_ids),
                    finish_reason=self.finish_reason_list[index],
                )
                for index in range(self.n)
            ],
            finished=True,
            prompt_logprobs=None,
        )


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_new_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_new_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(dtype)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)

    @classmethod
    def from_pretrained(cls, model_path: str, weight_path: str, server: RPCManager, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)

        cls.config = config
        cls.eos_token_ids = set()

        if hasattr(config, "eos_token_ids"):
            cls.eos_token_ids |= (
                set(config.eos_token_id) if isinstance(config.eos_token_id, list) else {config.eos_token_id}
            )

        cls.server = server
        cls.pp_size = len(cls.server.url_list)
        cls.tok = TokenizerUtils(model_path)
        if cls.tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(cls.tok.tokenizer.eos_token_id)

        state_dict = torch.load(weight_path)
        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.eval()
        return model

    def _prepare_forward_data(
        self, seq_input: SeqInput, hidden_states: torch.Tensor, need_serialize: bool
    ) -> Dict[str, Any]:
        if need_serialize:
            hidden_states = serialize_bfloat16_tensor(hidden_states)
        return {"uuid": seq_input.uuid_str_list, "seq_len": seq_input.seq_len_list, "hidden_states": hidden_states}

    def forward(self, inputs_embeds: torch.Tensor, seq_input: SeqInput) -> ForwardResult:
        hidden_states = inputs_embeds
        comm_cost_time_list = []
        last_pp_idx = self.pp_size - 1
        for pp_idx in range(self.pp_size):
            s1 = time.time()
            outputs = self.server.post_sync(
                pp_idx,
                "/forward",
                data=self._prepare_forward_data(seq_input, hidden_states, need_serialize=pp_idx == 0),
            )
            hidden_states = deserialize_bfloat16_tensor(outputs.output) if pp_idx == last_pp_idx else outputs.output
            s2 = time.time()
            comm_cost_time_list.append(s2 - s1 - outputs.cost_time)

        hidden_states = hidden_states.to(self.norm.weight.device)
        # hidden_states: bsz x seq_len x hidden_size
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # logits: bsz x seq_len x vocab_size
        # bsz: 1; seq_len: seq_len1 + seq_len2
        return ForwardResult(logits=logits, comm_cost_time_list=comm_cost_time_list)

    @torch.no_grad()
    async def generate(
        self,
        sequence_request_list: List[SequenceRequestData],
    ) -> AsyncGenerator:
        """
        @params:
            sequence_request_list: List[Params]
                Params:
                    input_ids: torch.Tensor

        """
        uuid_str_list, input_ids_list, seq_len_list = [], [], []
        for sequence_request in sequence_request_list:
            uuid_str_list.append(sequence_request.request_id)
            # 如果是 prefilling，则为 input_ids
            # 否则，为 output_ids[-1]
            if len(sequence_request.output_ids) == 0:
                input_ids_list.append(sequence_request.input_ids)
                seq_len_list.append(sequence_request.input_ids.shape[-1])
            else:
                input_ids_list.append(torch.tensor([sequence_request.output_ids[-1]]).unsqueeze(0))
                seq_len_list.append(1)

        input_ids = torch.cat(input_ids_list, dim=0)
        input_embeds = self.embed_tokens(input_ids)

        seq_input = SeqInput(uuid_str_list=uuid_str_list, seq_len_list=seq_len_list)
        forward_result = self(input_embeds, seq_input)
        logits = forward_result.logits

        # 根据 seq 拆开，之后直接在 sampler 中处理
        seq_logits_list = torch.split(logits, seq_input.seq_len_list, dim=1)
        for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
            generate_ids = sequence_request.sampler.decode(seq_logits)
            generate_texts = [self.tok.decode([x]) for x in generate_ids]

            sequence_request.output_ids.append(generate_ids[0])
            sequence_request.generate_ids = generate_ids
            sequence_request.generate_texts = generate_texts

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=self.eos_token_ids,
                max_new_tokens=sequence_request.sampling_params.get("max_new_tokens", 16),
            )
            if end.is_end:
                sequence_request.finish_reason_list = [end.finish_reason]
                sequence_request.is_stop = True

            sequence_request.output_text += generate_texts[0]  # 不添加 end text

            if len(sequence_request.output_ids) == 1:
                sequence_request.ttft_cost_time = forward_result.comm_cost_time_list
            else:
                sequence_request.tpot_cost_time = forward_result.comm_cost_time_list

        comm_cost_time_list = forward_result.comm_cost_time_list
        logging.info(f"communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")


class AsyncEngine:
    def __init__(self, model: MyLlamaForCausalLM):
        self.tok = model.tok
        self.model = model
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        self.limit_size: int = 5  # 每次最多处理 5 个请求，prefill + decode

    def preprocess(self, messages: List[Dict[str, Any]]) -> torch.Tensor:
        input_id_list = self.tok.preprocess(messages=messages).input_ids
        input_ids = torch.tensor(input_id_list).unsqueeze(0)
        return input_ids

    async def fetch_data(self):
        # prefill 队列和 decoding 队列的调度逻辑
        sequence_data_list = []

        # 优先从 decoding_queue 取数据
        while not self.decoding_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.decoding_queue.get()
            sequence_data_list.append(sequence_data)

        # 从 prefill_queue 中取数据，直到达到限制
        while not self.prefill_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.prefill_queue.get()
            sequence_data_list.append(sequence_data)

        return sequence_data_list

    async def _generate(self):
        while True:
            sequence_data_list: List[SequenceRequestData] = await self.fetch_data()
            if len(sequence_data_list) == 0:
                await asyncio.sleep(0.1)
                continue
            try:
                await self.model.generate(sequence_data_list)

                for sequence_data in sequence_data_list:
                    if not sequence_data.is_stop:
                        await self.decoding_queue.put(sequence_data)
                    async with sequence_data.condition:
                        sequence_data.condition.notify()

                # await asyncio.sleep(0.1)
            except Exception as e:
                logging.info(f"Error processing prefill_queue data: {str(e)}")
            except BaseException as e:
                logging.info(f"BaseException Error processing prefill_queue data: {str(e)}")
            finally:
                await asyncio.sleep(0.1)

    async def generate_stream(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
                    yield data.to_request_output()  # 流式返回数据的内容，可以控制

        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")

    async def generate(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
                    # 这里可以进行你需要的处理，例如更新输出
                    # 确保在这里将 output_text 更新
            return data.to_request_output()  # 返回最终的数据对象
        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")

    async def start(self):
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._generate())

    async def stop(self):
        logging.info("Stopping processing sequence_data")
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
