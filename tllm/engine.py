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
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return ForwardResult(logits=logits, comm_cost_time_list=comm_cost_time_list)

    def preprocess(self, messages: List[Dict[str, Any]]) -> torch.Tensor:
        input_id_list = self.tok.preprocess(messages=messages).input_ids
        input_ids = torch.tensor(input_id_list).unsqueeze(0)
        return input_ids

    @torch.no_grad()
    async def generate(
        self, input_ids: torch.Tensor, request_id: str, sampler: DecodeUtils, **kwargs
    ) -> AsyncGenerator:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        output_ids: List[int] = []
        output_text: str = ""
        finish_reason = None
        seq_len = input_embeds.shape[1]
        seq_input = SeqInput(uuid_str_list=[request_id], seq_len_list=[seq_len])
        while True:
            forward_result = self(input_embeds, seq_input)
            logits = forward_result.logits
            comm_cost_time_list = forward_result.comm_cost_time_list
            generate_ids = sampler.decode(logits)
            generate_texts = [self.tok.decode([x]) for x in generate_ids]
            output_ids.append(generate_ids[0])

            end = is_generate_end(output_ids, eos_token_ids=self.eos_token_ids, max_new_tokens=max_new_tokens)
            if end.is_end:
                finish_reason = end.finish_reason
                break

            output_text += generate_texts[0]  # 不添加 end text

            input_embeds = self.embed_tokens(torch.tensor(generate_ids)).unsqueeze(0)
            seq_input.seq_len_list = [1]
            if len(output_ids) == 1:
                logging.info(f"ttft communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")
            else:
                logging.info(f"tpot communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")

            await asyncio.sleep(0.1)
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=input_ids[0].tolist(),
                outputs=[
                    CompletionOutput(
                        index=0, text=generate_texts[0], token_ids=generate_ids[0], finish_reason=finish_reason
                    )
                ],
                finished=False,
                prompt_logprobs=None,
            )

        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=input_ids[0].tolist(),
            outputs=[CompletionOutput(index=0, text="", token_ids=tuple(output_ids), finish_reason=finish_reason)],
            finished=True,
            prompt_logprobs=None,
        )
