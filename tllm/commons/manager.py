import os
import time
from typing import List, Tuple

from transformers import AutoConfig

from tllm.commons.communicator import BaseCommunicator
from tllm.generate import LLMGenerator, TokenizerUtils
from tllm.models.file_helper import find_weight_file, get_model_path
from tllm.models.register import MODEL_REGISTER
from tllm.models.weight_helper import load_gguf_weight, read_from_safetensors, tie_embedding_weights


class WeightManager:
    def __init__(self, model_path: str):
        self.model_path = get_model_path(model_path)
        if str(self.model_path).endswith(".gguf"):
            self.read_master_weight = self._gguf_read_master_weight
            self.read_client_weight = self._gguf_read_client_weight
        else:
            self.read_master_weight = self._hf_read_master_weight
            self.read_client_weight = self._hf_read_client_weight
        self.state_dict = None
        self.tok, self.arch, self.config = self._post_init()

    def _post_init(self):
        if str(self.model_path).endswith(".gguf"):
            state_dict, config, _ = load_gguf_weight(str(self.model_path))
            if state_dict is None:
                raise ValueError("get gguf state_dict failed")
            self.state_dict = state_dict
            tok = TokenizerUtils("/Users/lujianghu/Documents/Llama-3.2-1B-Instruct/")  # TODO
            arch = "LlamaForCausalLM"
        else:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            tok = TokenizerUtils(self.model_path)
            arch = config.architectures[0]
        return tok, arch, config

    def _gguf_read_master_weight(self):
        new_state_dict = {}
        prefix_key_list = ["model.layers.", "rope_freqs"]
        for k, v in self.state_dict.items():
            flag = False
            for prefix_key in prefix_key_list:
                if k.startswith(prefix_key):
                    flag = True
                    break
            if flag:
                continue
            new_state_dict[k.split("model.")[-1]] = v
        state_dict = tie_embedding_weights(new_state_dict)
        if "norm.eps" not in state_dict:
            state_dict["norm.eps"] = self.config.rms_norm_eps
        return state_dict

    def _hf_read_weight(self, prefix_key_list: List[str]):
        file_set = find_weight_file(self.model_path, prefix_key_list)
        state_dict = {}
        for file in file_set:
            weight_path = os.path.join(self.model_path, file)
            state_dict.update(read_from_safetensors(weight_path, prefix_key_list))
        return state_dict

    def _hf_read_master_weight(self):
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
        state_dict = self._hf_read_weight(prefix_key_list)

        new_state_dict = {}
        for k, v in state_dict.items():
            # for qwen-vl
            if k == "visual.patch_embed.proj.weight":
                v = v.transpose(0, 2, 3, 4, 1)

            # model.layers for multi modal encoder
            if k.startswith("model.") and not k.startswith("model.layers."):
                new_state_dict[k.split("model.")[-1]] = v
            else:
                new_state_dict[k] = v

        state_dict = tie_embedding_weights(new_state_dict)
        return state_dict

    def _gguf_read_client_weight(self, start_idx: int, end_idx: int):
        if self.state_dict is None:
            raise ValueError("state_dict is None")
        new_state_dict = {}
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
        # TODO: support start_idx and end_idx
        for k, v in self.state_dict.items():
            flag = False
            for prefix_key in prefix_key_list:
                if k.startswith(prefix_key):
                    flag = True
                    break
            if flag:
                continue
            new_state_dict[k] = v
        return new_state_dict

    def _hf_read_client_weight(self, start_idx: int, end_idx: int):
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")

        prefix_key_list = [f"model.layers.{layer_idx}." for layer_idx in range(start_idx, end_idx)]
        state_dict = self._hf_read_weight(prefix_key_list)
        return state_dict


def load_client_model(start_idx: int, end_idx: int, comm: BaseCommunicator, model_path: str):
    weight_manager = WeightManager(model_path)
    state_dict = weight_manager.read_client_weight(start_idx, end_idx)
    if weight_manager.arch not in MODEL_REGISTER:
        raise ValueError(f"Model {weight_manager.arch} not supported")

    config = weight_manager.config

    config.comm = comm
    config.decoder_start_layer_idx = start_idx
    config.decoder_end_layer_idx = end_idx

    arch = config.architectures[0]

    _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

    s1 = time.perf_counter()
    model = MY_MODEL_CLASS.from_pretrained(config, model_path, state_dict)
    print(f"Model loaded in {time.perf_counter() - s1:.2f}s")
    return model


def load_master_model(model_path: str) -> Tuple[LLMGenerator, TokenizerUtils, int]:
    weight_manager = WeightManager(model_path)
    state_dict = weight_manager.read_master_weight()
    if weight_manager.arch not in MODEL_REGISTER:
        raise ValueError(f"Model {weight_manager.arch} not supported")

    MY_CausalLM_CLASS, _ = MODEL_REGISTER[weight_manager.arch]

    model = MY_CausalLM_CLASS.from_pretrained(weight_manager.config, model_path, state_dict)
    return model, weight_manager.tok, weight_manager.config.num_hidden_layers
