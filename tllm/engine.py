from dataclasses import dataclass
import logging
import time
from typing import *
import uuid

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tllm.commons.convert import deserialize_bfloat16_tensor, serialize_bfloat16_tensor
from tllm.generate.decode_utils import DecodeUtils
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
class ForwardResult:
    logits: torch.Tensor
    comm_cost_time_list: Optional[List[float]] = None
    hidden_states: Optional[torch.Tensor] = None


def is_generate_end(output_ids: List[int], eos_token_id: int, max_new_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_new_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] == eos_token_id:
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
        cls.server = server
        cls.pp_size = len(cls.server.url_list)

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

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, sampler: DecodeUtils, **kwargs) -> GenerateResult:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        output_ids: List[int] = []
        finish_reason = None
        uuid_str = str(uuid.uuid4())
        ttft_start_time, ttft_end_time = time.time(), time.time()

        seq_len = input_embeds.shape[1]
        seq_input = SeqInput(uuid_str_list=[uuid_str], seq_len_list=[seq_len])
        while True:
            forward_result = self(input_embeds, seq_input)
            logits = forward_result.logits
            comm_cost_time_list = forward_result.comm_cost_time_list
            generate_ids = sampler.decode(logits)
            output_ids.append(generate_ids[0])

            end = is_generate_end(output_ids, eos_token_id=self.config.eos_token_id, max_new_tokens=max_new_tokens)
            if end.is_end:
                finish_reason = end.finish_reason
                break

            input_embeds = self.embed_tokens(torch.tensor(generate_ids)).unsqueeze(0)
            seq_input.seq_len_list = [1]
            if len(output_ids) == 1:
                ttft_end_time = time.time()
                logging.info(f"ttft communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")
            else:
                logging.info(f"tpot communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")

        return GenerateResult(output_ids=output_ids, finish_reason=finish_reason, ttft=ttft_end_time - ttft_start_time)
