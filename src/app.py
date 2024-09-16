import argparse
import json
import time
import uuid
from typing import *

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from generate.decode_utils import DecodeUtils
from generate.token_utils import TokenizerUtils
from http_comm.server import Server
from rpc_comm.server import RPCServer
from utils import tensor_to_list

cost_time_dict = {}

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
        self.embedding.to(dtype=config.torch_dtype)
        self.lm_head.to(dtype=config.torch_dtype)
        self.norm.to(dtype=config.torch_dtype)

        self.load_model_flag = False

        self.server = server
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.config = config

    def init_client(self, tp_url_list: List[str], state_dict_path: str, layer_state_dict_dir: str, layer_start_end_list = None):
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
            if layer_start_end_list is not None:
                layer_idx_start, layer_start_end = layer_start_end_list[pp_idx]
            else:
                layer_idx_start, layer_start_end = pp_idx * step, (pp_idx + 1) * step            
            params_list.append(
                {
                    "config": self.config.to_dict(),
                    "layer_idx_start": layer_idx_start,
                    "layer_idx_end": layer_start_end,
                    "tp_url_list": tp_url_list,
                    "tp_size": self.tensor_parallel,
                    "layer_state_dict_dir": layer_state_dict_dir,
                }
            )

        response_list = self.server.post_thread("/init_model", params_list)
        assert self.server.is_success(response_list), "Model init failed"
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
        s1 = time.time()
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        cost_time_dict["embedding"] = time.time() - s1

        hidden_states = inputs_embeds
        for pp_idx in range(self.pipeline_parallel):
            s1 = time.time()
            # tp could request in parallel
            outputs = self.server.post_sync(
                pp_idx, "/forward", data=self._prepare_forward_data(uuid_str, hidden_states)
            )
            assert self.server.is_success(outputs), "Forward failed"
            hidden_states = self.server.fetch_list_output(outputs)
            cost_time_dict[f"forward {pp_idx}"] = time.time() - s1
            cost_time_dict[f"forward {pp_idx} calc"] = self.server.fetch_list_cost_time(outputs)

        s1 = time.time()
        hidden_states = self.norm(torch.tensor(hidden_states).to(inputs_embeds.dtype).to(self.norm.weight.device))
        logits = self.lm_head(hidden_states)
        cost_time_dict["lm_head"] = time.time() - s1
        return CausalLMOutputWithPast(logits=logits)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--comm", type=str, default="rpc", choices=["rpc", "http"])
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", required=False)
    parser.add_argument("--max-tokens", type=int, default=20, required=False)
    return parser.parse_args()


def test(llm, tok_path: str, text: str, max_tokens: int = 2):
    uuid_str = str(uuid.uuid4())

    tok_util = TokenizerUtils(tok_path)
    decode_util = DecodeUtils("greedy")
    text = "Who are you?"
    message = [{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}, {"role": "user", "content": text}]
    input_ids = tok_util.preprocess(message=message)
    print("input_ids: ", input_ids)

    formatted_prompt = "### Human: {}### Assistant:"
    text = formatted_prompt.format("Hello, how are you?")
    input_ids = tok_util.preprocess(text=text)
    output_ids = []
    output_texts = []
    for idx in range(1, max_tokens + 1):
        s1 = time.time()
        output = llm.forward(uuid_str, input_ids)
        print(f"cost time {idx}: {time.time() - s1:.2f} s")
        print("="*5 + f" cost time (detailed) " + "="*5)
        for k, v in cost_time_dict.items():
            print(f"{k}: {v:.2f} s")
        print("="*5 + f" cost time (detailed) " + "="*5)

        generate_ids = decode_util.decode(output.logits)
        generate_id = generate_ids[0]  # batch size = 1
        output_ids.append(generate_id)
        output_texts.append(tok_util.decode(generate_id))
        print(f"generate_id {idx}:", output_ids[-1], output_texts[-1])
        input_ids = torch.tensor(generate_ids).unsqueeze(0)
    print("output_texts: ", tok_util.tokenizer.decode(output_ids))

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
    llm = LLM(
        AutoConfig.from_pretrained(config["model_path"]),
        server,
        tensor_parallel=config["tensor_parallel"],
        pipeline_parallel=config["pipeline_parallel"],
    )
    if "layer_start_end_list" in config and len(config["layer_start_end_list"]) == config["pipeline_parallel"]:
        llm.init_client(config["tensor_parallel_url_list"], config["state_dict_path"], config["layer_state_dict_dir"], config["layer_start_end_list"])
    else:
        llm.init_client(config["tensor_parallel_url_list"], config["state_dict_path"], config["layer_state_dict_dir"])
    test(llm, config["model_path"], args.prompt, args.max_tokens)
