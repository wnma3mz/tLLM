import time
from typing import *
import uuid

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from rpc_comm.server import RPCServer
from src.utils import setup_seed, tensor_to_list, tokenize_message

# 使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行，通信时仅通信输入
# PYTHONPATH="./src2:./src":$PYTHONPATH python3 src2/app.py


class MyLlamaForCausalLM(nn.Module):
    # 主控制程序
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size

        url_list = ["localhost:50051", "localhost:25002"]
        self.pp_size = len(url_list)
        self.server = RPCServer(url_list)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)
        from transformers import LlamaForCausalLM

        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()

        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.eval()
        return model

    def _prepare_forward_data(self, uuid_str: str, hidden_states: torch.Tensor) -> Dict:
        return {"uuid": uuid_str, "hidden_states": tensor_to_list(hidden_states)}

    def forward(self, inputs_embeds: torch.Tensor, uuid_str: str):
        hidden_states = inputs_embeds
        for pp_idx in range(self.pp_size):
            outputs = self.server.post_sync(
                pp_idx,
                "/forward",
                data=self._prepare_forward_data(uuid_str, hidden_states),
            )
            assert self.server.is_success(outputs), "Forward failed"
            hidden_states = self.server.fetch_list_output(outputs)
            # TODO：发送一遍，接收一遍，相当于通信两遍。可以直接发送给下一个 PP
            # output = self.model.send(output, rank)

        hidden_states = torch.tensor(hidden_states).to(inputs_embeds.dtype).to(self.norm.weight.device)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, output.past_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[List[int], Optional[torch.Tensor]]:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        input_embeds = self.embed_tokens(input_ids)
        token_list: List[int] = []
        cnt = 0
        uuid_str = str(uuid.uuid4())
        while True:
            logits, past_key_values = self(input_embeds, uuid_str)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            next_token = torch.argmax(logits[:, -1], dim=1)
            token_list.append(next_token[0].tolist())
            print("token", token_list[-1])
            input_embeds = self.embed_tokens(next_token).unsqueeze(0)

        return token_list, None


def load_model_and_tokenizer(model_path: str) -> Tuple[MyLlamaForCausalLM, AutoTokenizer]:
    model = MyLlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


if __name__ == "__main__":
    # PP0（master）: decoder layer 第一部分 + embedding + lm head
    # PP: decoder layer 其他部分
    is_master = True
    setup_seed(42)

    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    s1 = time.time()
    model, tok = load_model_and_tokenizer(model_path)
    print(f"load_model cost time {time.time() - s1}")

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print(f"input_ids: {input_ids}")

    s1 = time.time()
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print(f"token: {output[0]}")
    print(f"cost time: {time.time() - s1}")

    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        print(f"Time taken: {time.time() - s1}")
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))
