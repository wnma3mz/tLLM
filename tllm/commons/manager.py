import time
from typing import Tuple

from transformers import AutoConfig

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.generate import LLMGenerator, TokenizerUtils
from tllm.models.register import MODEL_REGISTER


class ModelManager:
    def __init__(self, start_idx: int, end_idx: int):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def load_model(self, comm: SingleNodeCommunicator, model_path: str):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.comm = comm

        config.decoder_start_layer_idx = self.start_idx
        config.decoder_end_layer_idx = self.end_idx

        if model_path.endswith(".gguf"):
            arch = "MLXLlamaForCausalLM"
        else:
            arch = config.architectures[0]

            if arch not in MODEL_REGISTER:
                raise ValueError(f"Model {arch} not supported")

        _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

        # if model_path.endswith(".gguf"):
        #     weights, config, _ = load_gguf_weight(model_path)
        #     config.decoder_start_layer_idx = self.start_idx
        #     config.decoder_end_layer_idx = self.end_idx
        #     config.comm = SingleNodeCommunicator()
        s1 = time.perf_counter()
        model = MY_MODEL_CLASS.from_pretrained(config, model_path)
        print(f"Model loaded in {time.perf_counter() - s1:.2f}s")
        return model


def load_master_model(model_path: str, logger) -> Tuple[LLMGenerator, TokenizerUtils, int]:
    # if model_path.endswith(".gguf"):
    #     raise ValueError("GGUF model not supported")
    #     arch = "MLXLlamaForCausalLM"
    #     from tllm.models.gguf_utils import load_gguf_weight

    #     state_dict, config, _ = load_gguf_weight(model_path)
    #     tok_path = ...
    #     tok = TokenizerUtils(tok_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = config.architectures[0]
    if arch not in MODEL_REGISTER:
        raise ValueError(f"Model {arch} not supported")
    tok = TokenizerUtils(model_path)
    state_dict = None

    MY_CausalLM_CLASS, _ = MODEL_REGISTER[arch]

    model = MY_CausalLM_CLASS.from_pretrained(config, model_path, state_dict)
    return model, tok, config.num_hidden_layers
