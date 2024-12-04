# coding: utf-8
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, nn
from tinygrad.helpers import getenv

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
    assert (
        freqs_cis.shape[1] == xq.shape[1] == xk.shape[1]
    ), f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


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

    def __call__(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        attention_data: AttentionData,
    ) -> Tensor:
        if getenv("WQKV"):
            if not hasattr(self, "wqkv"):
                self.wqkv = Tensor.cat(self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
            xqkv = x @ self.wqkv.T
            xq, xk, xv = xqkv.split(
                [self.q_proj.weight.shape[0], self.k_proj.weight.shape[0], self.v_proj.weight.shape[0]], dim=2
            )
        else:
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        bsz, seqlen, _, _ = xq.shape

        if attention_data.request_cache is not None:
            # update the cache
            # assert xk.dtype == xv.dtype == cache.dtype, f"{xk.dtype=}, {xv.dtype=}, {cache.dtype=}"
            cache = attention_data.request_cache
            keys, values = cache.update_tinygrad(xk, xv, attention_data.uuid_list, self.layer_idx)
        else:
            keys = xk
            values = xv

        keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        attn = xq.scaled_dot_product_attention(keys, values, attention_data.attn_mask).transpose(1, 2)
        attn = attn.reshape(bsz, seqlen, -1)
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


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, num_layers: int) -> AttentionData:
    request_cache = RequestsCache(num_layers)
    q_len_list, k_len_list = [], [], []
    for uuid, q_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
        if uuid in cache_manager.cache_dict:
            # kv_cache 是整个历史的 kv_cache
            # 当 q_len 为 1 时，直接使用 kv_cache，使用历史的全部 token kv cache
            # TODO: 当 q_len > 1 时，表示只需要使用前 q_len 的 kv_cache，后面的 kv_cache 需要重新计算
            layer_cache_list, cache_seq_len = cache_manager.get(uuid)
            k_len_list.append(cache_seq_len + q_len)
        else:
            layer_cache_list = None
            k_len_list.append(q_len)
        q_len_list.append(q_len)
        request_cache.add(uuid, q_len, layer_cache_list)

    # TODO: support multi request
    start_pos = 0 if seq_input.seq_len_list[0] != 1 else seq_input.seq_len_list[0] - 1
    seqlen = seq_input.seq_len_list[0]
    mask = (
        Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-100000000")).triu(start_pos + 1).realize()
        if seqlen > 1
        else None
    )
    return AttentionData(
        request_cache=request_cache,
        attn_mask=mask,
        uuid_list=seq_input.uuid_list,
    )


def get_last_hidden_states(hidden_states: Tensor, seq_len_list: List[int]) -> Tensor:
    last_states = []
    current_idx = 0
    for seq_len in seq_len_list:
        sequence = hidden_states[current_idx : current_idx + seq_len]
        last_state = sequence[-1:]
        last_states.append(last_state)
        current_idx += seq_len
    return Tensor.cat(last_states, dim=0)


class TransformerBlock:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        max_context: int,
        linear=nn.Linear,
        feed_forward=FeedForward,
    ):
        self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)
        self.feed_forward = feed_forward(dim, hidden_dim, linear)
        self.attention_norm = nn.RMSNorm(dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, norm_eps)

    def __call__(self, x: Tensor, freqs_cis: Tensor, attention_data: AttentionData):
        h = x + self.attention(self.attention_norm(x), freqs_cis, attention_data)
        return (h + self.feed_forward(self.ffn_norm(h))).contiguous()


class TinyGradLlamaModel:
    def __init__(self, config, is_merge: bool = True, jit: bool = True):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.model = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx, is_merge)
        self.num_decoder_layers = config.decoder_end_layer_idx - config.decoder_start_layer_idx

        self.max_context = config.max_position_embeddings
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
            config.rope_theta,
            rope_scaling=config.rope_scaling,
        ).contiguous()
        self.forward_jit = TinyJit(self.forward) if jit else None

    def forward(self, hidden_states: Tensor, seq_input: SeqInput) -> Tensor:
        """
        @param hidden_states: bs x seq_len x hidden_size
        @param seq_input:
            uuid_list: List[str]: 每个请求的 uuid
            seq_len_list: List[int]: 每个请求的 seq_len
            如果 uuid 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: bs x seq_len x hidden_size
        """
        attention_data = build_forward_cache(seq_input, self.cache_manager, self.num_decoder_layers)

        attention_data.attn_mask = attention_data.attn_mask.to(hidden_states.device).to(hidden_states.dtype)
        start_pos = 0 if seq_input.seq_len_list[0] != 1 else seq_input.seq_len_list[0] - 1
        seqlen = seq_input.seq_len_list[0]
        freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))

        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = self.model(hidden_states, freqs_cis=freqs_cis, attention_data=attention_data)

        if self.config.decoder_end_layer_idx == self.config.num_hidden_layers:
            hidden_states = get_last_hidden_states(hidden_states, seq_input.seq_len_list)

        for uuid, seq_len in zip(seq_input.uuid_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid, attention_data.get_kv_cache_list(uuid), attention_data.get_cache_seq_len(uuid))
            self.cache_manager.check_alive()

        return hidden_states


class TinyGradLlamaForCausalLM:
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, x: np.ndarray) -> Tensor:
        return self.embed_tokens(Tensor(x, device=self.device))

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.to(self.dtype).to(self.norm.weight.device)
        # (seq_len1+seq_len2) x hidden_size
        logits = self.lm_head(self.norm(hidden_states)).realize()
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


def fix_bf16(weights: Dict[Any, Tensor]):
    if getenv("SUPPORT_BF16", 1):
        # TODO: without casting to float16, 70B llama OOM on tinybox.
        return {k: v.cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}
    # TODO: check if device supports bf16
    return {
        k: v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()
    }
