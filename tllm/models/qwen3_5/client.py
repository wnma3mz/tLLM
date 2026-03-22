from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import TextModelArgs as Qwen35TextModelArgs
from transformers import AutoConfig

from tllm.models.backend_mlx.helper import MLXCacheManager, quantization_func
from tllm.models.qwen3_5.decoder import Qwen35Decoder
from tllm.schemas import SeqInput

cache_manager = MLXCacheManager()


class MLXQwen35Model(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        config_dict = config.to_dict()
        text_config = config_dict.get("text_config", config_dict)
        comm = config.comm
        del config.comm

        args = Qwen35TextModelArgs.from_dict(text_config)
        args.comm = comm
        self.rank = args.comm.rank
        self.world_size = args.comm.world_size

        self.config = config
        self.model = Qwen35Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

        # qwen3.5 uses mixed cache (ssm + kv), only keep sequence length in request cache.
        cache_manager.init_request_cache(self.num_layers, -1, -1, -1)
        is_start_pp = self.config.decoder_start_layer_idx == 0
        is_end_pp = self.config.decoder_end_layer_idx == self.config.num_hidden_layers
        cache_manager.post_init(is_start_pp, is_end_pp)

    @classmethod
    def from_pretrained(cls, config: AutoConfig, state_dict: Dict[str, mx.array], **kwargs):
        model = cls(config)
        state_dict = model.sanitize(state_dict, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()))
        mx.eval(model.parameters())
        model.eval()
        return model

    @staticmethod
    def sanitize(weights, start_idx: int, end_idx: int):
        sanitized_weights = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                k = k.replace("language_model.", "")
            if not k.startswith("model.layers."):
                continue
            if int(k.split("model.layers.", 1)[-1].split(".")[0]) not in range(start_idx, end_idx):
                continue
            sanitized_weights[k] = v
        return sanitized_weights

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        hidden_states = cache_manager.build_forward_cache(hidden_states, seq_input)
        output = self.model(hidden_states, cache=cache_manager.attn_data, cache_manager=cache_manager)

        # TODO 异步更新 cache
        cache_manager.update_cache(seq_input)
        output = cache_manager.get_last_hidden_states(output)
        return output
