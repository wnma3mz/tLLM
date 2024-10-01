from dataclasses import dataclass
from typing import *
import uuid

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tllm.generate.decode_utils import DecodeUtils
from tllm.rpc.manager import RPCManager
from tllm.utils import tensor_to_list

finish_reason_type = Literal["length", "stop", None]


@dataclass
class GenerateResult:
    output_ids: List[int]
    finish_reason: Optional[finish_reason_type] = None
    output_text: Optional[str] = None


@dataclass
class GenerateEnd:
    finish_reason: finish_reason_type
    is_end: bool


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
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

    def _prepare_forward_data(self, uuid_str: str, hidden_states: torch.Tensor) -> Dict[str, Any]:
        return {"uuid": uuid_str, "hidden_states": tensor_to_list(hidden_states)}

    def forward(self, inputs_embeds: torch.Tensor, uuid_str: str) -> Tuple[torch.Tensor, None]:
        hidden_states = inputs_embeds
        for pp_idx in range(self.pp_size):
            outputs = self.server.post_sync(
                pp_idx, "/forward", data=self._prepare_forward_data(uuid_str, hidden_states)
            )
            assert self.server.is_success(outputs), "Forward failed"
            hidden_states = self.server.fetch_list_output(outputs)

        hidden_states = torch.tensor(hidden_states).to(inputs_embeds.dtype).to(self.norm.weight.device)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, None

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, sampler: DecodeUtils, **kwargs) -> GenerateResult:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        output_ids: List[int] = []
        finish_reason = None
        uuid_str = str(uuid.uuid4())
        while True:
            logits, _ = self(input_embeds, uuid_str)
            generate_ids = sampler.decode(logits)
            output_ids.append(generate_ids[0])

            end = is_generate_end(output_ids, eos_token_id=self.config.eos_token_id, max_new_tokens=max_new_tokens)
            if end.is_end:
                finish_reason = end.finish_reason
                break

            input_embeds = self.embed_tokens(torch.tensor(generate_ids)).unsqueeze(0)

        return GenerateResult(output_ids=output_ids, finish_reason=finish_reason)
