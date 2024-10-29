import time
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

# 使用 torch.dist 实现 张量并行，通信时仅通信输入
# export OMP_NUM_THREADS=8; torchrun --nproc_per_node=2 benchmarks/torch_dist_model.py


def setup_seed(seed):
    torch.manual_seed(seed)


def is_rank0():
    return dist.get_rank() == 0


def print_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)


class Communicator:
    def __init__(self) -> None:
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def all_reduce(self, x: torch.Tensor):
        # input shape == output shape
        # output = torch.sum(torch.stack(input), dim=0)
        # each node get output
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor):
        cluster_output = [torch.zeros_like(x, dtype=x.dtype) for _ in range(self.world_size)]
        dist.all_gather(cluster_output, x)
        return torch.cat(cluster_output, dim=-1)

    def gather(self, x: torch.Tensor):
        # 只在节点 0 上聚合
        cluster_output = (
            [torch.zeros_like(x, dtype=x.dtype) for _ in range(self.world_size)] if self.rank == 0 else None
        )
        dist.gather(x, gather_list=cluster_output, dst=0)
        return torch.cat(cluster_output, dim=-1) if self.rank == 0 else None


class ColumnParallelLayer(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert col_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=False)

    def load_weight(self, w: Optional[torch.Tensor] = None):
        w_list = w.chunk(self.world_size, dim=0)
        w = w_list[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        return node_output


class MergeParallelLayer(nn.Module):
    def __init__(self, row_size: int, col_size: int, dup_layer: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert col_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.dup_layer = dup_layer
        self.layer = nn.Linear(row_size, col_size * self.dup_layer // self.world_size, bias=False)

    def load_weight(self, w_list: Optional[List[torch.Tensor]] = None):
        w = w_list[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        return torch.chunk(node_output, self.dup_layer, dim=-1)


class QKVParallelLayer(nn.Module):
    def __init__(self, row_size: int, col_size_list: List[int]) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        for x in col_size_list:
            assert x % self.world_size == 0
        col_size = sum(col_size_list)
        assert col_size % self.world_size == 0

        self.row_size, self.col_size = row_size, col_size
        self.col_size_list = [x // self.world_size for x in col_size_list]
        self.layer = nn.Linear(row_size, col_size // self.world_size, bias=False)

    def load_weight(self, w_list: Optional[List[torch.Tensor]] = None):
        w = w_list[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_output = self.layer(x)
        return torch.split(node_output, self.col_size_list, dim=-1)


class RowParallelLayer(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert row_size % self.world_size == 0
        self.row_size, self.col_size = row_size, col_size
        self.layer = nn.Linear(row_size // self.world_size, col_size, bias=False)

    def load_weight(self, w: Optional[torch.Tensor] = None):
        w_list = w.chunk(self.world_size, dim=1)
        w = w_list[self.rank]
        self.load_state_dict({"layer.weight": w})

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_x = torch.chunk(x, self.world_size, dim=-1)[self.rank]
        node_output = self.layer(node_x)
        return node_output

    @torch.no_grad()
    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MyLlamaMLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # self.gate_proj = ColumnParallelLayer(self.hidden_size, self.intermediate_size)
        # self.up_proj = ColumnParallelLayer(self.hidden_size, self.intermediate_size)
        self.gate_up_proj = MergeParallelLayer(self.hidden_size, self.intermediate_size, 2)
        self.down_proj = RowParallelLayer(self.intermediate_size, self.hidden_size)

    def load_state_dict(self, state_dict: Dict) -> None:
        # for key in ["gate_proj", "up_proj", "down_proj"]:
        for key in ["down_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.mlp.{key}.weight"
            getattr(self, key).load_weight(state_dict.get(layer_name, None))

        weight_chunks = []
        len_ = dist.get_world_size()
        for key in ["gate_proj", "up_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.mlp.{key}.weight"
            weight = state_dict[layer_name]
            weight_chunks.append(torch.chunk(weight, len_, dim=0))
        combined_weights = [torch.cat([chunk[i] for chunk in weight_chunks], dim=0) for i in range(len_)]
        self.gate_up_proj.load_weight(combined_weights)

    def forward(self, x):
        gate_out, up_out = self.gate_up_proj(x)
        intermediate_states = self.act_fn(gate_out) * up_out
        return comm.all_reduce(self.down_proj.forward_chunk(intermediate_states))


class MyLlamaSdpaAttention(nn.Module):
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
        self.world_size = dist.get_world_size()
        # self.q_proj = ColumnParallelLayer(self.hidden_size, self.num_heads * self.head_dim)
        # self.k_proj = ColumnParallelLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        # self.v_proj = ColumnParallelLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.qkv_proj = QKVParallelLayer(
            self.hidden_size,
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
        )
        self.o_proj = RowParallelLayer(self.num_heads * self.head_dim, self.hidden_size)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def load_state_dict(self, state_dict: Dict) -> None:
        # for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        for key in ["o_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.self_attn.{key}.weight"
            getattr(self, key).load_weight(state_dict.get(layer_name, None))

        weight_chunks = []
        len_ = dist.get_world_size()
        for key in ["q_proj", "k_proj", "v_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.self_attn.{key}.weight"
            weight = state_dict[layer_name]
            weight_chunks.append(torch.chunk(weight, len_, dim=0))
        combined_weights = [torch.cat([chunk[i] for chunk in weight_chunks], dim=0) for i in range(len_)]
        self.qkv_proj.load_weight(combined_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = comm.all_reduce(self.o_proj.forward_chunk(attn_output))
        return attn_output, None, past_key_value


class MyLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MyLlamaSdpaAttention(config=config, layer_idx=layer_idx)

        self.mlp = MyLlamaMLP(config, layer_idx=layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def load_state_dict(self, state_dict: Dict):
        # 用两次 layer norm 的时间替换两次通信的时间
        self.input_layernorm.load_state_dict(
            {"weight": state_dict.pop(f"model.layers.{self.layer_idx}.input_layernorm.weight")}
        )
        self.post_attention_layernorm.load_state_dict(
            {"weight": state_dict.pop(f"model.layers.{self.layer_idx}.post_attention_layernorm.weight")}
        )

        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
    ) -> Tuple[torch.Tensor, Optional["Cache"]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class MyLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.decoder = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def load_state_dict(self, state_dict: Dict) -> None:
        for layer in self.decoder:
            layer.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["Cache"] = None,
    ):
        next_decoder_cache = None
        for layer in self.decoder:
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values,
            )
            hidden_states = layer_outputs[0]

            # 所有层的 kv cache 放到一起了，所以这里只需要取最后一层的 kv cache
            next_decoder_cache = layer_outputs[1]
        next_cache = next_decoder_cache
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = MyLlamaModel(config)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)
        from transformers import LlamaForCausalLM

        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()

        if is_rank0():
            model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
            model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
            model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.model.load_state_dict(state_dict)

        model.eval()
        return model

    def forward(self, input_embeds: torch.Tensor, position_ids, past_key_values):
        output = self.model(input_embeds, position_ids, past_key_values)
        if is_rank0():
            hidden_states = self.norm(output.last_hidden_state)
            logits = self.lm_head(hidden_states)
            return logits, output.past_key_values
        return None, output.past_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[List[int], Optional[torch.Tensor]]:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        bs, seq_len = input_ids.size()  # bs == 1
        past_key_values = None
        position_ids = None
        input_embeds = self.embed_tokens(input_ids)
        token_list: List[int] = []
        cnt = 0
        while True:
            dist.broadcast(input_embeds, src=0)
            # print(f"Rank: {dist.get_rank()}, input_embeds: {input_embeds}")
            if past_key_values is None:
                past_key_values = DynamicCache()
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            else:
                past_key_values_length = past_key_values.get_seq_length()
                position_ids = torch.arange(past_key_values_length, 1 + past_key_values_length, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0)

            logits, past_key_values = self(input_embeds, position_ids, past_key_values)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            if is_rank0():
                next_token = torch.argmax(logits[:, -1], dim=1)
                token_list.append(next_token[0].tolist())
                input_embeds = self.embed_tokens(next_token).unsqueeze(0)
            else:
                # 必须要加上这一行
                input_embeds = self.embed_tokens(torch.tensor([0])).unsqueeze(0)

        return token_list, None


def load_model_and_tokenizer(
    model_path: str,
) -> Tuple[MyLlamaForCausalLM, AutoTokenizer]:
    model = MyLlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


formatted_prompt = "### Human: {}### Assistant:"


def tokenize_message(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> List[int]:
    inputs = formatted_prompt.format(messages[0]["content"])
    # inputs = "Hello, how are you?"
    # inputs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(inputs, add_special_tokens=True)
    while input_ids[0] == input_ids[1] == tok.bos_token_id:
        # input_ids = input_ids[1:]
        input_ids.pop(0)
    return input_ids


if __name__ == "__main__":
    setup_seed(42)
    # 初始化分布式环境
    dist.init_process_group(backend="gloo")
    comm = Communicator()

    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    s1 = time.time()
    model, tok = load_model_and_tokenizer(model_path)
    print_rank0(f"load_model cost time {time.time() - s1}")

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print_rank0("input_ids: ", input_ids)
    # output = model.generate(input_ids, max_new_tokens=50, tokenizer=tok, eos_token_id=[0, tok.eos_token_id])
    # print(tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))

    s1 = time.time()
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print_rank0("token: ", output[0])
    print_rank0("Cost time: ", time.time() - s1)

    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        print_rank0(f"Time taken: {time.time() - s1}")
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))
