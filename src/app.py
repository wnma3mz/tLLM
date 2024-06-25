import time
import uuid
from typing import *

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from communication.server import Server
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

    def init_client(self, tp_url_list: List[str], state_dict_path: str):
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
                    "state_dict_path": state_dict_path,
                }
            )

        response_list = self.server.post_thread("/init_model", params_list)
        for response in response_list:
            assert response.status_code == 200, "Model init failed"
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
            if outputs.status_code != 200:
                raise Exception("Forward failed")
            output_data = outputs.json()["output"]
            # 只需要更新 hidden_states
            hidden_states = output_data["last_hidden_state"]

        hidden_states = self.norm(list_to_tensor(hidden_states).to(self.norm.weight.device))
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits)


if __name__ == "__main__":
    pipeline_parallel_url_list = ["http://localhost:8000", "http://localhost:8001"]
    tensor_parallel_url_list = ["http://localhost:8002", "http://localhost:8003"]
    tensor_parallel, pipeline_parallel = len(tensor_parallel_url_list), len(pipeline_parallel_url_list)
    # tensor_parallel, tensor_parallel_url_list = 1, ["http://localhost:8002"]
    # pipeline_parallel, pipeline_parallel_url_list = 1, ["http://localhost:8000"]

    model_name = "TinyLlama-1.1B-Chat-v1.0"
    model_path = f"/Users/lujianghu/Documents/{model_name}"
    server = Server(pipeline_parallel_url_list)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    llm = LLM(config, server, tensor_parallel=tensor_parallel, pipeline_parallel=pipeline_parallel)

    state_dict_path = "./weights/TinyLlama-1.1B-Chat-v1.0.pth"
    llm.init_client(tensor_parallel_url_list, state_dict_path)

    uuid_str = str(uuid.uuid4())
    input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")
    print("input_ids: ", input_ids)
    idx = 1
    s1 = time.time()
    output = llm.forward(uuid_str, input_ids)
    print(f"cost time {idx}: {time.time() - s1:.2f} s")

    generate_id = torch.argmax(output.logits[0, -1], 0)
    print(f"generate_id {idx}:", generate_id, tokenizer.decode(generate_id.tolist()))
    input_ids = torch.cat((input_ids, generate_id.unsqueeze(0).unsqueeze(0)), dim=1)
    idx += 1
    output = llm.forward(uuid_str, input_ids)
    generate_id = torch.argmax(output.logits[0, -1], 0)
    print(f"cost time {idx}: {time.time() - s1:.2f} s")
    print(f"generate_id {idx}:", generate_id, tokenizer.decode(generate_id.tolist()))
