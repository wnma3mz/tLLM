import time
from typing import *

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 使用 ray 实现 张量并行，通信时通信权重和输入，以复用 ray 的 actor

ray.init(ignore_reinit_error=True, num_cpus=4)


@ray.remote
class ParallelWeightLinear(nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)


linear_actor = ParallelWeightLinear.remote()


@ray.remote
def compute_linear(x, weight):
    return ray.get(linear_actor.forward.remote(x, weight))


class MyLlamaMLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        assert config.intermediate_size % config.tp == 0
        self.tp = config.tp
        self.slice_size = config.intermediate_size // config.tp
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

    def load_state_dict(self, state_dict: Dict) -> None:
        self.w = {}
        for key in ["gate_proj", "up_proj", "down_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.mlp.{key}.weight"
            if key == "down_proj":
                w_list = state_dict[layer_name].chunk(self.tp, dim=1)
            else:
                w_list = state_dict[layer_name].chunk(self.tp, dim=0)
            self.w[key] = w_list

    def forward(self, x):
        gate_futures = [compute_linear.remote(x, self.w["gate_proj"][i]) for i in range(self.tp)]
        up_futures = [compute_linear.remote(x, self.w["up_proj"][i]) for i in range(self.tp)]

        gate_results = torch.cat(ray.get(gate_futures), dim=-1)
        gate_out = self.act_fn(gate_results)

        up_out = torch.cat(ray.get(up_futures), dim=-1)
        intermediate_states = gate_out * up_out

        split_x_list = torch.chunk(intermediate_states, self.tp, dim=-1)
        down_futures = [compute_linear.remote(split_x_list[i], self.w["down_proj"][i]) for i in range(self.tp)]
        results = ray.get(down_futures)
        return torch.sum(torch.stack(results), dim=0)


class MyLlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.tp = config.tp
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
        assert (self.num_heads * self.head_dim) % self.tp == 0
        assert (self.num_key_value_heads * self.head_dim) % self.tp == 0

        self.query_out_slices = self.num_heads * self.head_dim // self.tp
        self.key_value_slicing = self.num_key_value_heads * self.head_dim // self.tp

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def load_state_dict(self, state_dict: Dict) -> None:
        self.w = {}
        for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            layer_name = f"model.layers.{self.layer_idx}.self_attn.{key}.weight"
            if key[0] in "qkv":
                w_list = state_dict[layer_name].chunk(self.tp, dim=0)
            else:
                w_list = state_dict[layer_name].chunk(self.tp, dim=1)
            self.w[key] = w_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q_futures = [compute_linear.remote(hidden_states, self.w["q_proj"][i]) for i in range(self.tp)]
        k_futures = [compute_linear.remote(hidden_states, self.w["k_proj"][i]) for i in range(self.tp)]
        v_futures = [compute_linear.remote(hidden_states, self.w["v_proj"][i]) for i in range(self.tp)]

        query_states = torch.cat(ray.get(q_futures), dim=-1)
        key_states = torch.cat(ray.get(k_futures), dim=-1)
        value_states = torch.cat(ray.get(v_futures), dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        split_x_list = torch.chunk(attn_output, self.tp, dim=-1)
        futures = [compute_linear.remote(split_x_list[i], self.w["o_proj"][i]) for i in range(self.tp)]
        results = ray.get(futures)
        attn_output = torch.sum(torch.stack(results), dim=0)

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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        config.tp = 4
        # config.num_hidden_layers = 1

        model = cls(config)
        from transformers import LlamaForCausalLM

        state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()
        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.model.load_state_dict(state_dict)

        model.eval()
        return model

    def forward(self, input_embeds: torch.Tensor, position_ids, past_key_values):
        output = self.model(input_embeds, position_ids, past_key_values)
        hidden_states = self.norm(output.last_hidden_state)
        logits = self.lm_head(hidden_states)
        return logits, output.past_key_values

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # input_ids: bs x seq_len
        max_new_tokens = kwargs.get("max_new_tokens", 16)
        bs, seq_len = input_ids.size()  # bs == 1
        past_key_values = None
        position_ids = None
        input_embeds = self.embed_tokens(input_ids)
        token_list = []
        cnt = 0
        while True:
            if past_key_values is None:
                past_key_values = DynamicCache()
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            else:
                past_key_values_length = past_key_values.get_seq_length()
                position_ids = torch.arange(past_key_values_length, 1 + past_key_values_length, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0)

            logits, past_key_values = self(input_embeds, position_ids, past_key_values)
            next_token = torch.argmax(logits[:, -1], dim=1)
            token_list.append(next_token)
            cnt += 1
            if cnt >= max_new_tokens:
                break
            input_embeds = self.embed_tokens(next_token).unsqueeze(0)

        return token_list, None


def load_model_and_tokenizer(model_path: str) -> Tuple[MyLlamaForCausalLM, AutoTokenizer]:
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
    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    model, tok = load_model_and_tokenizer(model_path)

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print("input_ids: ", input_ids)
    # output = model.generate(input_ids, max_new_tokens=50, tokenizer=tok, eos_token_id=[0, tok.eos_token_id])
    # print(tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))

    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    token_list = output[0]
    # print("token: ", token_list)
    # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))

    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        print(f"Time taken: {time.time() - s1}")
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))
