import asyncio
from dataclasses import dataclass
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
    input_ids: List[int]
    output_ids: List[int]
    output_text: str
    finish_reason: str
    sampler: DecodeUtils

    ttft_cost_time: List[float]
    tpot_cost_time: List[float]
    timeout: int = 100000  # 请求的总超时时间
    sampling_params: Dict = {}
    is_stop: bool = False


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
    async def generate_bak(
        self,
        params_list: List[SequenceRequestData],
    ) -> AsyncGenerator:
        # max_new_tokens = params.sampling_params.get("max_new_tokens", 16)
        input_ids = torch.cat([params.input_ids for params in params_list], dim=0)
        # input_embeds = self.embed_tokens(input_ids)
        # output_ids: List[int] = []
        # output_text: str = ""
        # finish_reason = None
        # # seq_len = input_embeds.shape[1]
        # seq_input = SeqInput(uuid_str_list=[params.request_id for params in params_list], seq_len_list=[len(params.input_ids) for params in params_list])
        # while True:
        #     forward_result = self(input_embeds, seq_input)
        #     logits = forward_result.logits
        #     # 根据 seq 拆开，之后直接在 sampler 中处理
        #     torch.split(logits, [], dim=1)

        #     comm_cost_time_list = forward_result.comm_cost_time_list
        #     # generate_ids = sampler.decode(logits)
        #     generate_texts = [self.tok.decode([x]) for x in generate_ids]
        #     output_ids.append(generate_ids[0])

        #     end = is_generate_end(output_ids, eos_token_ids=self.eos_token_ids, max_new_tokens=max_new_tokens)
        #     if end.is_end:
        #         finish_reason = end.finish_reason
        #         break

        #     output_text += generate_texts[0]  # 不添加 end text

        #     input_embeds = self.embed_tokens(torch.tensor(generate_ids)).unsqueeze(0)
        #     seq_input.seq_len_list = [1] # TODO
        #     if len(output_ids) == 1:
        #         logging.info(f"ttft communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")
        #     else:
        #         logging.info(f"tpot communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")

        #     await asyncio.sleep(0.1)
        #     yield RequestOutput(
        #         request_id=request_id,
        #         prompt=None,
        #         prompt_token_ids=input_ids[0].tolist(),
        #         outputs=[
        #             CompletionOutput(
        #                 index=0, text=generate_texts[0], token_ids=generate_ids[0], finish_reason=finish_reason
        #             )
        #         ],
        #         finished=False,
        #         prompt_logprobs=None,
        #     )

        # yield RequestOutput(
        #     request_id=request_id,
        #     prompt=None,
        #     prompt_token_ids=input_ids[0].tolist(),
        #     outputs=[CompletionOutput(index=0, text="", token_ids=tuple(output_ids), finish_reason=finish_reason)],
        #     finished=True,
        #     prompt_logprobs=None,
        # )

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
        input_ids = torch.cat([sequence_request.input_ids for sequence_request in sequence_request_list], dim=0)
        input_embeds = self.embed_tokens(input_ids)
        # seq_len = input_embeds.shape[1]
        seq_input = SeqInput(
            uuid_str_list=[sequence_request.request_id for sequence_request in sequence_request_list],
            seq_len_list=[len(sequence_request.input_ids) for sequence_request in sequence_request_list],
        )

        forward_result = self(input_embeds, seq_input)
        logits = forward_result.logits

        # 根据 seq 拆开，之后直接在 sampler 中处理
        seq_logits_list = torch.split(logits, seq_input.seq_len_list, dim=1)
        for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
            generate_ids = sequence_request.sampler.decode(seq_logits)
            generate_texts = [self.tok.decode([x]) for x in generate_ids]

            if sequence_request.output_ids:
                sequence_request.output_ids.append(generate_ids[0])
            else:
                sequence_request.output_ids = [generate_ids[0]]

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=self.eos_token_ids,
                max_new_tokens=sequence_request.sampling_params.get("max_new_tokens", 16),
            )
            if end.is_end:
                sequence_request.finish_reason = end.finish_reason
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
        self.queue: asyncio.Queue = asyncio.Queue()
        self.decode_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        # 存储任务的Future对象
        self.results: Dict[str, asyncio.Future] = {}
        self.limit_size: int = 5  # 每次最多处理 5 个请求，prefill + decode

    def preprocess(self, messages: List[Dict[str, Any]]) -> torch.Tensor:
        input_id_list = self.tok.preprocess(messages=messages).input_ids
        input_ids = torch.tensor(input_id_list).unsqueeze(0)
        return input_ids

    def is_stop(self, result):
        if result.finished:
            return True
        return False

    async def fetch_tasks(self):
        tasks = []

        # 优先从 decode_queue 取数据
        while len(tasks) < self.limit_size:
            try:
                request_id, item = await self.decode_queue.get()
                tasks.append((request_id, item))
                self.decode_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 从 queue 中取数据，直到达到限制
        while len(tasks) < self.limit_size:
            try:
                request_id, item = await self.queue.get()
                tasks.append((request_id, item))
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        return tasks

    async def _generate(self):
        while True:
            try:
                tasks = await self.fetch_tasks()

                result = await self.model.generate(tasks)
                if not self.is_stop(result):
                    await self.decode_queue.put((task_id, data))
                else:
                    if task_id in self.results:
                        self.results[task_id].set_result(result)
                self.queue.task_done()
            except Exception as e:
                logging.info(f"Error processing queue data: {e}")
                if task_id in self.results:
                    self.results[task_id].set_exception(e)
            finally:
                if task_id in self.results and self.results[task_id].done():
                    del self.results[task_id]
            await asyncio.sleep(1)

    async def generate(self, request_id: str, data: SequenceRequestData):
        try:
            self.results[request_id] = asyncio.Future()
            await self.queue.put((request_id, data))

            result = await asyncio.wait_for(self.results[request_id], data.timeout)
            return result
        except asyncio.TimeoutError:
            if request_id in self.results:
                del self.results[request_id]
            raise TimeoutError("Processing timed out")
        except Exception as e:
            if request_id in self.results:
                del self.results[request_id]
            raise e

    async def start(self):
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._generate())

    async def stop(self):
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
