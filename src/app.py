import argparse
import json
import time
import uuid
from typing import *

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from http_comm.server import Server
from rpc_comm.server import RPCServer
from rpc_comm.convert import protobuf_to_list
from utils import list_to_tensor, tensor_to_list


class LLM:
    """
    localhost:
    - tokenizer
    - embedding
    - send to decoder
    - receive from decoder
    - lm_head
    - de tokenizer
    """

    def __init__(self, config, server, tensor_parallel: int = 1, pipeline_parallel: int = 1):
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.load_model_flag = False

        self.server = server
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.config = config

    def init_client(self, tp_url_list: List[str], state_dict_path: str, layer_state_dict_dir: str):
        if self.load_model_flag:
            print(f"Model has been initialized")
            return
        # TP need more check
        assert self.config.num_hidden_layers % self.pipeline_parallel == 0
        assert self.config.hidden_size % self.tensor_parallel == 0
        assert self.config.intermediate_size % self.tensor_parallel == 0
        assert len(tp_url_list) == self.tensor_parallel

        print("Model initializing...")
        s1 = time.time()
        # localhost init model
        state_dict = torch.load(state_dict_path, "cpu")
        self.embedding.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})
        self.lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})
        self.norm.load_state_dict({"weight": state_dict["model.norm.weight"]})

        # remote init model
        step = self.config.num_hidden_layers // self.pipeline_parallel
        params_list = []
        for pp_idx in range(self.pipeline_parallel):
            params_list.append(
                {
                    "config": self.config.to_dict(),
                    "layer_idx_start": pp_idx * step,
                    "layer_idx_end": (pp_idx + 1) * step,
                    "tp_url_list": tp_url_list,
                    "tp_size": self.tensor_parallel,
                    "layer_state_dict_dir": layer_state_dict_dir,
                }
            )

        response_list = self.server.post_thread("/init_model", params_list)
        for response in response_list:
            assert response.status == 200, "Model init failed"
        print(f"Model initialized cost time: {time.time() - s1:.2f} s")
        self.load_model_flag = True

    def _prepare_forward_data(self, uuid_str: str, hidden_states: torch.Tensor) -> Dict:
        return {"uuid": uuid_str, "hidden_states": tensor_to_list(hidden_states)}

    def forward(
        self,
        uuid_str: str,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        hidden_states = inputs_embeds
        for pp_idx in range(self.pipeline_parallel):
            # tp could request in parallel
            outputs = self.server.post_sync(
                pp_idx, "/forward", data=self._prepare_forward_data(uuid_str, hidden_states)
            )
            if outputs.status != 200:
                raise Exception("Forward failed")
            hidden_states = protobuf_to_list(outputs.output)
            # output_data = outputs.json()["output"]
            # # 只需要更新 hidden_states
            # hidden_states = output_data["last_hidden_state"]

        hidden_states = self.norm(list_to_tensor(hidden_states).to(self.norm.weight.device))
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--comm", type=str, default="rpc", choices=["rpc", "http"])
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", required=False)
    parser.add_argument("--max-tokens", type=int, default=2, required=False)
    return parser.parse_args()


def test(llm, tokenizer, text: str, max_tokens: int = 2):
    uuid_str = str(uuid.uuid4())
    input_ids = tokenizer.encode(text, return_tensors="pt")
    print("input_ids: ", input_ids)

    for idx in range(1, max_tokens + 1):
        s1 = time.time()
        output = llm.forward(uuid_str, input_ids)
        print(f"cost time {idx}: {time.time() - s1:.2f} s")

        generate_id = torch.argmax(output.logits[0, -1], 0)
        print(f"generate_id {idx}:", generate_id, tokenizer.decode(generate_id.tolist()))
        input_ids = torch.cat((input_ids, generate_id.unsqueeze(0).unsqueeze(0)), dim=1)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)

    if args.comm == "rpc":
        server = RPCServer(config["pipeline_parallel_url_list"])
    elif args.comm == "http":
        server = Server(config["pipeline_parallel_url_list"])
    else:
        raise Exception("Comm type not supported")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True, use_fast=False)
    llm = LLM(
        AutoConfig.from_pretrained(config["model_path"]),
        server,
        tensor_parallel=config["tensor_parallel"],
        pipeline_parallel=config["pipeline_parallel"],
    )
    llm.init_client(config["tensor_parallel_url_list"], config["state_dict_path"], config["layer_state_dict_dir"])

    test(llm, tokenizer, args.prompt, args.max_tokens)
