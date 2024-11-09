import glob
import itertools
import math
import os
import re
from typing import *

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import ModelArgs
import numpy as np
from transformers import AutoConfig

from tllm.commons.mlx_layers import MyTransformerBlock
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.cache import AttentionData, CacheManager, RequestsCache
from tllm.models.protocol import SeqInput
from tllm.models.utils import load_master_weight


def build_mlx_mask(seq_len_list: List[Tuple[int, int]], total_L: int, total_S: int) -> mx.array:
    mask_list = [
        mx.tril(mx.ones((L, S), dtype=mx.bool_), k=0) if L > 1 else mx.ones((L, S), dtype=mx.bool_)
        for (L, S) in seq_len_list
    ]

    combined_mask = mx.zeros((total_L, total_S), dtype=mx.bool_)

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.shape[0], r_index : r_index + mask.shape[1]] = mask
        l_index += mask.shape[0]
        r_index += mask.shape[1]

    final_mask = mx.where(combined_mask, 0, -math.inf)
    return final_mask


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    actual_seq_len_list = []
    L, S = 0, 0
    for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            actual_seq_len_list.append([q_len, cache_seq_len + q_len])
            L += q_len
            S += cache_seq_len + q_len
        else:
            layer_cache_list = None
            actual_seq_len_list.append([q_len, q_len])
            L += q_len
            S += q_len
        request_cache.add(uuid, q_len, layer_cache_list)
    return AttentionData(
        request_cache=request_cache,
        attn_mask=build_mlx_mask(actual_seq_len_list, L, S),
        uuid_list=seq_input.uuid_list,
    )


def quantization_func(config, model, state_dict):
    if getattr(config, "quantization", None) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in state_dict

        nn.quantize(
            model,
            **config.quantization,
            class_predicate=class_predicate,
        )
    else:
        model.set_dtype(mx.bfloat16)

    return model


def empty_func(h, mask, cache):
    # TODO
    return h


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layers = [empty_func] * start_layer_idx + [
            MyTransformerBlock(args=args, layer_idx=layer_idx, offset=start_layer_idx, is_merge=is_merge)
            for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, h: mx.array, mask, cache: AttentionData) -> mx.array:
        for layer in self.layers:
            h = layer(h, mask, cache=cache)
        return h


class MyMLXLlamaModel(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        args = ModelArgs.from_dict(config.to_dict())
        self.vocab_size = args.vocab_size
        self.cache_manager = CacheManager()
        self.args = args
        self.config = config
        is_merge = getattr(config, "is_merge", True)
        self.model = Decoder(args, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

    def __call__(self, hidden_states: mx.array, seq_input: SeqInput) -> mx.array:
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_layers)

        mask = attention_data.attn_mask
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = self.model(hidden_states, mask=mask, cache=attention_data)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()
        return output

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, config: AutoConfig, model_path: str, state_dict: Optional[Any] = None):
        model = cls(config)
        if state_dict is None:
            weights = model.read_weight_from_model_path(model_path)
        else:
            weights = state_dict

        model = quantization_func(config, model, weights)
        model.load_weights(list(weights.items()), strict=False)
        if getattr(config, "quantization", None) is None:
            model.set_dtype(mx.bfloat16)

        mx.eval(model.parameters())
        model.eval()
        return model

    def read_weight_from_model_path(self, model_path: str) -> Dict[str, mx.array]:
        # Not Support quantization
        print(f"start_idx: {self.config.decoder_start_layer_idx}, end_idx: {self.config.decoder_end_layer_idx}")
        attn_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn")
        mlp_layer_idx_pattern = re.compile(r"model\.layers\.(\d+)\.mlp")
        qkv_proj_list = ["q_proj", "k_proj", "v_proj"]
        gate_up_list = ["gate_proj", "up_proj"]
        weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]

        prefix_key_list += [
            f"model.layers.{i}."
            for i in range(self.config.num_hidden_layers)
            if not (self.config.decoder_start_layer_idx <= i < self.config.decoder_end_layer_idx)
        ]
        key_list = list(weights.keys())
        for key in key_list:
            for prefix_key in prefix_key_list:
                if key.startswith(prefix_key):
                    weights.pop(key)

        key_list = list(weights.keys())

        attn_proj_w = {}  # layer_idx -> {qkv: weight}
        mlp_w = {}
        for key in key_list:
            attn_res = attn_layer_idx_pattern.findall(key)
            mlp_res = mlp_layer_idx_pattern.findall(key)
            if attn_res:
                layer_idx = int(attn_res[0])
                if layer_idx not in attn_proj_w:
                    attn_proj_w[layer_idx] = {}
            elif mlp_res:
                layer_idx = int(mlp_res[0])
                if layer_idx not in mlp_w:
                    mlp_w[layer_idx] = {}
            else:
                continue

            for qkv in qkv_proj_list:
                if qkv in key:
                    attn_proj_w[layer_idx].update({qkv: weights.pop(key)})
            for mlp in gate_up_list:
                if mlp in key:
                    mlp_w[layer_idx].update({mlp: weights.pop(key)})

            layer_weights = attn_proj_w.get(layer_idx, [])
            if len(layer_weights) == 3:
                name = f"model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
                weights[name] = mx.concatenate([layer_weights[qkv] for qkv in qkv_proj_list], axis=0)
                attn_proj_w.pop(layer_idx)

            layer_weights = mlp_w.get(layer_idx, [])
            if len(layer_weights) == 2:
                name = f"model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
                weights[name] = mx.concatenate([layer_weights[mlp] for mlp in gate_up_list], axis=0)
                mlp_w.pop(layer_idx)

        # TODO: support bias and TP
        return weights


class MyMLXLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dtype = mx.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, logger, config, tok: TokenizerUtils, model_path: str, state_dict: Optional[Any] = None):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.logger = logger
        cls.eos_token_ids = set()

        if hasattr(config, "eos_token_ids"):
            if isinstance(config.eos_token_id, list):
                cls.eos_token_ids |= set(config.eos_token_ids)
            else:
                cls.eos_token_ids.add(config.eos_token_id)

        if tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(tok.tokenizer.eos_token_id)
        eos_token = tok.tokenizer.decode(list(cls.eos_token_ids))
        cls.logger.debug(f"eos_token_ids: {cls.eos_token_ids}; Tokens: {eos_token}")

        if state_dict is None:
            state_dict = load_master_weight(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.") and not k.startswith("model.layers."):
                new_state_dict[k.split("model.")[-1]] = v
        state_dict = new_state_dict
        has_key_list = list(state_dict.keys())
        if "lm_head.weight" not in state_dict:
            for key in has_key_list:
                if key.startswith("embed_tokens."):
                    state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]

        model = quantization_func(config, model, state_dict)
        model.load_weights(list(state_dict.items()), strict=False)

        mx.eval(model.parameters())
        model.eval()
        return model

    def get_input_embeddings(self, x: np.ndarray) -> mx.array:
        return self.embed_tokens(mx.array(x))

    def get_logits(self, hidden_states: mx.array, seq_len_list: List[int]) -> List[mx.array]:
        # 只取最后一个 token 的 hidden_states
        index_list = list(itertools.accumulate(seq_len_list[:-1]))
        seq_hidden_states = mx.split(hidden_states, index_list, axis=0)
        hidden_states = mx.concat([x[-1:, :] for x in seq_hidden_states], axis=0).astype(self.dtype)
        logits = self.lm_head(self.norm(hidden_states))
        ind = list(itertools.accumulate([1] * (len(seq_len_list) - 1)))
        return mx.split(logits, ind, axis=0)
