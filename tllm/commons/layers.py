from typing import *
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
)

from tllm.commons.attn import get_attention_implementation
from tllm.commons.cache import AttentionData, RequestsCache

# Get the best available attention implementation
self_attn_func, attention_type = get_attention_implementation()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)


class BaseParallelLayer(nn.Module):
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        super().__init__()


class MergeParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size: int, dup_layer: int, world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        assert col_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.dup_layer = dup_layer
        self.layer = nn.Linear(row_size, col_size * self.dup_layer // self.world_size, bias=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        node_output = self.layer(x)
        return torch.chunk(node_output, self.dup_layer, dim=-1)


class QKVParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size_list: List[int], world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        for x in col_size_list:
            assert x % self.world_size == 0
        col_size = sum(col_size_list)
        assert col_size % self.world_size == 0

        self.row_size, self.col_size = row_size, col_size
        self.col_size_list = [x // self.world_size for x in col_size_list]
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        node_output = self.layer(x)
        return torch.split(node_output, self.col_size_list, dim=-1)


class RowParallelLayer(BaseParallelLayer):
    def __init__(self, row_size: int, col_size: int, world_size: int, rank: int) -> None:
        super().__init__(world_size, rank)
        assert row_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.layer = nn.Linear(row_size // self.world_size, col_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MergedLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.comm = config.comm
        self.rank = self.comm.rank
        self.world_size = self.comm.world_size

        self.gate_up_proj = MergeParallelLayer(self.hidden_size, self.intermediate_size, 2, self.world_size, self.rank)
        self.down_proj = RowParallelLayer(self.intermediate_size, self.hidden_size, self.world_size, self.rank)

    def forward(self, x):
        # x: [seq_len, hidden_size]
        gate_out, up_out = self.gate_up_proj(x)
        return self.comm.all_reduce(self.down_proj(self.act_fn(gate_out) * up_out))


class MergedLlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.comm = config.comm
        self.rank = self.comm.rank
        self.world_size = self.comm.world_size

        self.qkv_proj = QKVParallelLayer(
            self.hidden_size,
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            self.world_size,
            self.rank,
        )
        self.o_proj = RowParallelLayer(self.num_heads * self.head_dim, self.hidden_size, self.world_size, self.rank)

        max_seq_len = 1024
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self._k_cache = torch.zeros(size=(max_seq_len, self.num_key_value_heads, self.head_dim), dtype=torch.bfloat16, device=self.device)
        # self._v_cache = torch.zeros(size=(max_seq_len, self.num_key_value_heads, self.head_dim), dtype=torch.bfloat16, device=self.device)
        self._k_cache, self._v_cache = None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data: AttentionData,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [seq_len, hidden_size]
        q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        query_states = query_states.view(q_len, -1, self.head_dim)
        key_states = key_states.view(q_len, -1, self.head_dim)
        value_states = value_states.view(q_len, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, attention_data.position_ids, unsqueeze_dim=1
        )
        request_cache: RequestsCache = attention_data.request_cache
        key_states, value_states = request_cache.update(
            key_states,
            value_states,
            attention_data.uuid_list,
            self.layer_idx - self.config.offset,
            self._k_cache,
            self._v_cache,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self_attn_func(query_states, key_states, value_states, attn_mask=attention_data.attn_mask)

        attn_output = attn_output.reshape(q_len, -1)

        attn_output = self.comm.all_reduce(self.o_proj(attn_output))
        return attn_output, None


class PlainLlamaSdpaAttention(LlamaSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data: AttentionData,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [seq_len, hidden_size]
        q_len, _ = hidden_states.size()

        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.view(q_len, -1, self.head_dim).transpose(0, 1)
        key_states = key_states.view(q_len, -1, self.head_dim).transpose(0, 1)
        value_states = value_states.view(q_len, -1, self.head_dim).transpose(0, 1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, attention_data.position_ids, unsqueeze_dim=0
        )
        request_cache: RequestsCache = attention_data.request_cache
        key_states, value_states = request_cache.update(
            key_states, value_states, attention_data.uuid_list, self.layer_idx - self.config.offset
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self_attn_func(query_states, key_states, value_states, attn_mask=attention_data.attn_mask)

        attn_output = attn_output.reshape(q_len, -1)

        return self.o_proj(attn_output), None


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, is_merge: bool) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        if is_merge:
            self.mlp = MergedLlamaMLP(config)
            self.self_attn = MergedLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        else:
            self.mlp = LlamaMLP(config)
            self.self_attn = PlainLlamaSdpaAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_data: AttentionData,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_data=attention_data,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
