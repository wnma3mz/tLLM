# coding: utf-8
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad.nn.state import load_state_dict, safe_load, torch_load

from tllm.commons.cache import AttentionData, CacheManager, RequestsCache
from tllm.schemas import SeqInput

# Edited from https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py


# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, dtype=dtypes.half, rope_scaling: Optional[Dict[str, float]] = None
) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[: (dim // 2)] / dim))

    if rope_scaling:
        factor = rope_scaling.get("factor", 1.0)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 1.0)
        original_max_pos_emb = rope_scaling.get("original_max_position_embeddings", end)

        freqs[: dim // 4] *= low_freq_factor
        freqs[dim // 4 :] = freqs[dim // 4 :].contiguous() * high_freq_factor
        freqs *= (original_max_pos_emb / end) ** (1.0 / factor)

    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    # TODO: move dtype outside this
    return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim // 2, 2)


# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
    a, b = A[..., 0:1], A[..., 1:2]
    ro = a * c - b * d
    co = a * d + b * c
    return ro.cat(co, dim=-1)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    # check seq len
    assert (
        freqs_cis.shape[0] == xq.shape[0] == xk.shape[0]
    ), f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 4
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(2), xk_out.flatten(2)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
    return x.repeat((1, 1, n_rep)).reshape(seqlen, n_kv_heads * n_rep, head_dim)


class Attention:
    def __init__(self, config, layer_idx: Optional[int] = None, linear=nn.Linear):
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _rope(self, xq, xk, freqs_cis: List, request_cache: RequestsCache, uuid_list: List[str]):
        xq_list, xk_list = [], []
        start = 0
        import copy

        for uuid, fc in zip(uuid_list, freqs_cis):
            end = start + request_cache.get_seq_len(uuid)
            xq_, xk_ = apply_rotary_emb(copy.deepcopy(xq[start:end]), copy.deepcopy(xk[start:end]), fc)
            xq_list.append(xq_)
            xk_list.append(xk_)
            start = end
        return xq_list[0].cat(*xq_list[1:], dim=0), xk_list[0].cat(*xk_list[1:], dim=0)

    def __call__(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_data: AttentionData,
    ) -> Tensor:
        # bsz x seq_len x hidden_size
        L, _ = x.shape
        if getenv("WQKV"):
            if not hasattr(self, "wqkv"):
                self.wqkv = Tensor.cat(self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
            xqkv = x @ self.wqkv.T
            xq, xk, xv = xqkv.split(
                [self.q_proj.weight.shape[0], self.k_proj.weight.shape[0], self.v_proj.weight.shape[0]], dim=1
            )
        else:
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.reshape(L, self.num_heads, self.head_dim)
        xk = xk.reshape(L, self.num_key_value_heads, self.head_dim)
        xv = xv.reshape(L, self.num_key_value_heads, self.head_dim)

        xq, xk = self._rope(xq, xk, freqs_cis, attention_data.request_cache, attention_data.uuid_list)

        seqlen, _, _ = xq.shape

        cache = attention_data.request_cache
        keys, values = cache.update_tinygrad(xk, xv, attention_data.uuid_list, self.layer_idx)

        keys, values = repeat_kv(keys, self.num_key_value_groups), repeat_kv(values, self.num_key_value_groups)
        xq, keys, values = xq.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0)
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        attn = xq.scaled_dot_product_attention(keys, values, attention_data.attn_mask).transpose(1, 2)
        attn = attn[0].reshape(seqlen, -1)
        return self.o_proj(attn)


class FeedForward:
    def __init__(self, config, linear=nn.Linear):
        self.config = config

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = linear(self.intermediate_size, self.hidden_size, bias=False)  # the gate in Gated Linear Unit

    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))  # SwiGLU [arxiv/2002.05202, eq (5)]


def build_tinygrad_mask(q_len_list: List[int], k_len_list: List[int]) -> Tensor:
    mask_list = [
        Tensor.ones((L, S)).triu(1) if L > 1 else Tensor.zeros((L, S)) for (L, S) in zip(q_len_list, k_len_list)
    ]

    combined_mask = Tensor.zeros((sum(q_len_list), sum(k_len_list))).contiguous()

    l_index, r_index = 0, 0
    for mask in mask_list:
        combined_mask[l_index : l_index + mask.shape[0], r_index : r_index + mask.shape[1]] = mask
        l_index += mask.shape[0]
        r_index += mask.shape[1]

    final_mask = Tensor.where(combined_mask, float("-inf"), 0).realize()
    return final_mask


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    q_len_list, k_len_list = [], []
    for uuid, q_len in zip(seq_input.uuid_list, ...):
        if uuid in cache_manager.cache_dict:
            layer_cache_list = cache_manager.get(uuid)
            cache_seq_len = ...
            k_len_list.append(cache_seq_len + q_len)
        else:
            layer_cache_list = None
            k_len_list.append(q_len)
        q_len_list.append(q_len)
        request_cache.add(uuid, q_len, layer_cache_list)

    return AttentionData(
        request_cache=request_cache,
        attn_mask=build_tinygrad_mask(q_len_list, k_len_list),
        uuid_list=seq_input.uuid_list,
        position_ids=[q_len_list, k_len_list],
    )


def get_last_hidden_states(hidden_states: Tensor, seq_len_list: List[int]) -> Tensor:
    last_states = []
    current_idx = 0
    for seq_len in seq_len_list:
        sequence = hidden_states[current_idx : current_idx + seq_len]
        last_state = sequence[-1:]
        last_states.append(last_state)
        current_idx += seq_len
    return last_states[0].cat(*last_states[1:], dim=0)


class TransformerBlock:
    def __init__(self, config, layer_idx: int, is_merge: bool):
        self.self_attn = Attention(config, layer_idx)
        self.mlp = FeedForward(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: Tensor, freqs_cis: Tensor, attention_data: AttentionData):
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis, attention_data)
        return (h + self.mlp(self.post_attention_layernorm(h))).contiguous()


class TinyGradLlamaModel:
    def __init__(self, config, is_merge: bool = True, jit: bool = True):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

        self.max_context = config.max_position_embeddings
        self.forward_jit = TinyJit(self.forward) if jit else None
        self._freqs_cis = None

    @property
    def freqs_cis(self):
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.config.hidden_size // self.config.num_attention_heads,
                self.config.max_position_embeddings * 2,
                self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            ).contiguous()[0]
        return self._freqs_cis

    @classmethod
    def from_pretrained(cls, config, model_path: str, state_dict: Optional[Dict] = None, is_merge: bool = True):
        model = cls(config)

        if (model_path / "model.safetensors.index.json").exists():
            weights = load(str(model_path / "model.safetensors.index.json"))
        elif (model_path / "model.safetensors").exists():
            weights = load(str(model_path / "model.safetensors"))
        else:
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")

        # TODO: 切 PP
        state_dict = weights
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("model.layers.")}

        # Only Tiny grad
        def permute(v: Tensor, n_heads: int):
            return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

        key_list = list(weights.keys())
        for k in key_list:
            if "q_proj" in k:
                state_dict[k] = permute(state_dict[k], model.config.num_attention_heads)
            elif "k_proj" in k:
                state_dict[k] = permute(state_dict[k], model.config.num_key_value_heads)

        state_dict = fix_bf16(state_dict)
        load_state_dict(model, state_dict)

        return model

    def forward(self, hidden_states: Tensor, seq_input: SeqInput) -> Tensor:
        """
        @param hidden_states: bs x seq_len x hidden_size
        @param seq_input:
            uuid_list: List[str]: 每个请求的 uuid
            seq_len_list: List[int]: 每个请求的 seq_len
            如果 uuid 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: bs x seq_len x hidden_size
        """
        # Not support multi requests
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_decoder_layers)

        if attention_data.attn_mask is not None:
            attention_data.attn_mask = attention_data.attn_mask.cast(hidden_states.dtype).to(hidden_states.device)
        q_len_list, k_len_list = attention_data.position_ids

        freqs_cis = []
        for q_len, k_len in zip(q_len_list, k_len_list):
            tmp_freqs_cis = self.freqs_cis.cast(hidden_states.dtype).realize()
            tmp_freqs_cis = tmp_freqs_cis.shrink(((k_len - q_len, k_len), None, None, None))
            freqs_cis.append(tmp_freqs_cis)

        hidden_states = self.model(hidden_states, freqs_cis=freqs_cis, attention_data=attention_data)

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            split_len_list = attention_data.q_len_list
            hidden_states = get_last_hidden_states(hidden_states, split_len_list)

        for uuid in seq_input.uuid_list:
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid))
            self.cache_manager.check_alive()

        return hidden_states

    def __call__(self, hidden_states, seq_input):
        return self.forward(hidden_states, seq_input).realize()


def load(fn: str):
    if fn.endswith(".index.json"):
        with open(fn) as fp:
            weight_map = json.load(fp)["weight_map"]
        parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
        return {k: parts[n][k] for k, n in weight_map.items()}
    elif fn.endswith(".gguf"):
        gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
        from tinygrad.nn.state import gguf_load

        return gguf_load(gguf_tensor)[1]
    elif fn.endswith(".safetensors"):
        return safe_load(fn)
    else:
        return torch_load(fn)


class TinyGradLlamaForCausalLM:
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, config, model_path: str, state_dict: Optional[Dict] = None, is_merge: bool = True):
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers

        if (model_path / "model.safetensors.index.json").exists():
            state_dict = load(str(model_path / "model.safetensors.index.json"))
        elif (model_path / "model.safetensors").exists():
            state_dict = load(str(model_path / "model.safetensors"))
        else:
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")

        state_dict = {k.split("model.")[-1]: v for k, v in state_dict.items() if not k.startswith("model.layers.")}
        has_key_list = list(state_dict.keys())
        if "lm_head.weight" not in state_dict:
            for key in has_key_list:
                if key.startswith("embed_tokens."):
                    state_dict[key.replace("embed_tokens.", "lm_head.")] = state_dict[key]
        state_dict = fix_bf16(state_dict)
        load_state_dict(model, state_dict)

        return model

    def get_input_embeddings(self, x: np.ndarray) -> Tensor:
        # bs x seq_len x hidden_size -> seq_len x hidden_size
        return self.embed_tokens(Tensor(x, device=self.embed_tokens.weight.device))[0]

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.cast(self.norm.weight.dtype).to(self.norm.weight.device)
        # (seq_len1+seq_len2) x hidden_size
        logits = self.lm_head(self.norm(hidden_states))
        return logits


class EmptyLayer:
    def __call__(self, hidden_states: Tensor, freqs_cis, attention_data) -> Tensor:
        return hidden_states


class Decoder:
    def __init__(self, config, start_layer_idx: int, end_layer_idx: int, is_merge: bool):
        config.offset = start_layer_idx
        self.layers = [
            TransformerBlock(config, layer_idx, is_merge) for layer_idx in range(start_layer_idx, end_layer_idx)
        ]

    def __call__(self, hidden_states: Tensor, freqs_cis, attention_data: AttentionData) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis=freqs_cis, attention_data=attention_data)
        return hidden_states


def fix_bf16(weights: Dict[Any, Tensor]) -> Dict[Any, Tensor]:
    if getenv("SUPPORT_BF16", 1):
        # TODO: without casting to float16, 70B llama OOM on tinybox.
        # return {k: v.cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}
        return weights
    # TODO: check if device supports bf16
    return {
        k: v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()
    }
