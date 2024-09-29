from typing import *
import uuid

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tllm.rpc.manager import RPCManager
from tllm.utils import tensor_to_list


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
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[List[int], None]:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        token_list: List[int] = []
        cnt = 0
        uuid_str = str(uuid.uuid4())
        while True:
            logits, _ = self(input_embeds, uuid_str)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            next_token = torch.argmax(logits[:, -1], dim=1)
            token_list.append(next_token[0].tolist())
            print("token", token_list[-1])
            input_embeds = self.embed_tokens(next_token).unsqueeze(0)

        return token_list, None
