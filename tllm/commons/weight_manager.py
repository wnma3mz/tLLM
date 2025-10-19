import glob
import os

from transformers import AutoConfig

from tllm.commons.tp_communicator import BaseCommunicator
from tllm.models.file_helper import get_model_path
from tllm.models.register import DEP_MODEL_REGISTER, MODEL_REGISTER
from tllm.models.utils import read_from_text_config
from tllm.models.weight_helper import load_gguf_weight, read_from_safetensors


class WeightManager:
    def __init__(self, model_path: str):
        self.model_path = get_model_path(model_path)
        self.state_dict = None

        if "flux" in str(self.model_path).lower():
            from mflux import ModelConfig

            print("load flux model weight")
            self.read_master_weight = self._read_flux_master_weight
            self.read_client_weight = self._read_flux_client_weight

            if "schnell" in str(self.model_path).lower():
                config = ModelConfig.from_alias("schnell")
            elif "dev" in str(self.model_path).lower():
                config = ModelConfig.from_alias("dev")
            else:
                raise ValueError("ModelConfig not found")

            config.num_hidden_layers = 38
            config.model_name = str(self.model_path)
            self.config = config

            self.tok, self.arch = None, "FLUX"
        else:
            if str(self.model_path).endswith(".gguf"):
                self.read_master_weight = self._gguf_read_master_weight
                self.read_client_weight = self._gguf_read_client_weight
            else:
                self.read_master_weight = self._hf_read_weight
                self.read_client_weight = self._hf_read_weight
            self.tok, self.arch, self.config = self._post_init()

    def _read_flux_master_weight(self):
        from mflux.weights.weight_handler import WeightHandler

        weights = WeightHandler(local_path=self.model_path, lora_paths=None, lora_scales=None)

        self.config.quantization_level = weights.quantization_level
        return weights

    def _read_flux_client_weight(self, start_idx: int, end_idx: int):
        from mflux.weights.weight_handler import WeightHandler

        weights, self.config.quantization_level = WeightHandler.load_transformer(root_path=self.model_path)

        class TransformerWeightHandler:
            def __init__(self, weights):
                self.transformer = weights

        return TransformerWeightHandler(weights)

    def _post_init(self):
        from tllm.generate import TokenizerUtils

        if str(self.model_path).endswith(".gguf"):
            raise NotImplementedError("GGUF model not supported")
            # state_dict, config, _ = load_gguf_weight(str(self.model_path))
            # if state_dict is None:
            #     raise ValueError("get gguf state_dict failed")
            # self.state_dict = state_dict
            # tok = TokenizerUtils(...)  # TODO
            # arch = "LlamaForCausalLM"
        else:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            tok = TokenizerUtils(self.model_path)
            arch = config.architectures[0]

        assert hasattr(config, "num_hidden_layers") or hasattr(getattr(config, "text_config"), "num_hidden_layers")
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
        state_dict = new_state_dict
        if "norm.eps" not in state_dict:
            state_dict["norm.eps"] = self.config.rms_norm_eps
        return state_dict

    def _hf_read_weight(self):
        sf_file_list = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        state_dict = {}
        for sf_file in sf_file_list:
            state_dict.update(read_from_safetensors(sf_file))
        return state_dict

    def _gguf_read_client_weight(self):
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


def load_client_model(start_idx: int, end_idx: int, comm: BaseCommunicator, model_path: str):
    weight_manager = WeightManager(model_path)
    config = weight_manager.config

    if getattr(config, "num_hidden_layers", None) is None:
        num_hidden_layers = read_from_text_config(config, "num_hidden_layers")
        quantization = getattr(config, "quantization", None)
        config = config.text_config
        config.quantization = quantization
    else:
        num_hidden_layers = config.num_hidden_layers

    end_idx = min(end_idx, num_hidden_layers)

    state_dict = weight_manager.read_client_weight()
    if weight_manager.arch not in MODEL_REGISTER:
        raise ValueError(f"Model {weight_manager.arch} not supported")

    config.comm = comm
    config.decoder_start_layer_idx = start_idx
    config.decoder_end_layer_idx = end_idx

    _, MY_MODEL_CLASS = MODEL_REGISTER[weight_manager.arch]

    kwargs = {}
    if weight_manager.arch == "FLUX":
        kwargs.update({"quantization_level": weight_manager.config.quantization_level})

    model = MY_MODEL_CLASS.from_pretrained(config, state_dict, **kwargs)
    return model


def load_master_model(model_path: str):
    weight_manager = WeightManager(model_path)
    state_dict = weight_manager.read_master_weight()
    if weight_manager.arch not in MODEL_REGISTER:
        arch = weight_manager.arch
        if weight_manager.arch in DEP_MODEL_REGISTER:
            raise ValueError(f"Model {arch} now is support, please execute `pip install {DEP_MODEL_REGISTER[arch]}`")
        else:
            raise ValueError(f"Model {arch} not supported")

    MY_CausalLM_CLASS, _ = MODEL_REGISTER[weight_manager.arch]

    kwargs = {}
    kwargs.update({"model_path": weight_manager.model_path})
    if weight_manager.arch == "FLUX":
        kwargs.update({"quantization_level": weight_manager.config.quantization_level})

    model = MY_CausalLM_CLASS.from_pretrained(weight_manager.config, state_dict, **kwargs)
    model.tok = weight_manager.tok
    return model
