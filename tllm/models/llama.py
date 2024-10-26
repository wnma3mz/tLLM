import time
from typing import *

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from tllm.commons.cache_manager import CacheManager
from tllm.commons.convert import deserialize_bfloat16_tensor, serialize_bfloat16_tensor
from tllm.commons.layers import AttentionCache, MyLlamaDecoderLayer
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.cache import SeqDynamicCache
from tllm.models.protocol import ForwardResult, SequenceRequestData
from tllm.models.utils import build_mask, is_generate_end
from tllm.rpc.manager import RPCManager
from tllm.rpc.protocol import SeqInput


def build_forward_cache(seq_input: SeqInput, cache_manager: CacheManager, device: str) -> AttentionCache:
    position_ids_list = []
    past_key_values = SeqDynamicCache()
    actual_seq_len_list = []
    for uuid_str, seq_len in zip(seq_input.uuid_str_list, seq_input.seq_len_list):
        if uuid_str in cache_manager.cache_dict:
            kv_cache = cache_manager.get(uuid_str)
            position_ids = torch.tensor([kv_cache.get_seq_length()], dtype=torch.long).unsqueeze(0)
            actual_seq_len_list.append([seq_len, kv_cache.get_seq_length() + 1])
        else:
            kv_cache = None
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            actual_seq_len_list.append([seq_len, seq_len])
        past_key_values.add(uuid_str, seq_len, kv_cache)
        position_ids_list.append(position_ids)

    return AttentionCache(
        position_ids=torch.cat(position_ids_list, dim=-1).to(device),
        past_key_value=past_key_values,
        uuid_str_list=seq_input.uuid_str_list,
        attn_mask=build_mask(actual_seq_len_list),
    )


class Decoder(nn.Module):
    def __init__(self, config, start_layer_idx: int, end_layer_idx: int):
        super().__init__()
        config.offset = start_layer_idx
        self.decoder = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(start_layer_idx, end_layer_idx)]
        )

    def load_state_dict(self, state_dict: Dict) -> None:
        for layer in self.decoder:
            layer.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_cache: AttentionCache,
    ):
        next_decoder_cache = None
        for i, layer in enumerate(self.decoder):
            layer_outputs = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_cache=attention_cache,
            )
            hidden_states = layer_outputs[0]

            # 所有层的 kv cache 放到一起了，所以这里只需要取最后一层的 kv cache
            next_decoder_cache = layer_outputs[1]
        next_cache = next_decoder_cache
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


class MyLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.cache_manager = CacheManager()
        self.config = config
        self.decoder = Decoder(config, config.decoder_start_layer_idx, config.decoder_end_layer_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decoder.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, seq_input: SeqInput) -> torch.Tensor:
        """
        @param hidden_states: bs x seq_len x hidden_size
        @param seq_input:
            uuid_str_list: List[str]: 每个请求的 uuid
            seq_len_list: List[int]: 每个请求的 seq_len
            如果 uuid_str 存在，则使用缓存的 kv cache，否则使用新的 kv cache

        @return: bs x seq_len x hidden_size
        """
        attention_cache = build_forward_cache(seq_input, self.cache_manager, self.device)
        hidden_states = hidden_states.to(self.device)
        position_embeddings = self.rotary_emb(hidden_states, attention_cache.position_ids)
        output = self.decoder(hidden_states, position_embeddings=position_embeddings, attention_cache=attention_cache)

        for uuid_str, seq_len in zip(seq_input.uuid_str_list, seq_input.seq_len_list):
            self.cache_manager.set(uuid_str, output.past_key_values.get_cache(uuid_str))
            self.cache_manager.check_alive()
        return output.last_hidden_state

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        dtype = torch.bfloat16
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(dtype)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype)

    @classmethod
    def from_pretrained(cls, logger, model_path: str, weight_path: str, server: RPCManager, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)

        cls.config = config
        cls.num_layers = config.num_hidden_layers
        cls.logger = logger
        cls.eos_token_ids = set()

        if hasattr(config, "eos_token_ids"):
            cls.eos_token_ids |= (
                set(config.eos_token_id) if isinstance(config.eos_token_id, list) else {config.eos_token_id}
            )

        cls.server = server
        cls.pp_size = len(cls.server.url_list)
        cls.tok = TokenizerUtils(model_path)
        if cls.tok.tokenizer.eos_token_id:
            cls.eos_token_ids.add(cls.tok.tokenizer.eos_token_id)
        # cls.logger.debug("eos_token_ids", cls.eos_token_ids, cls.tok.tokenizer.convert_ids_to_tokens(list(cls.eos_token_ids)))

        state_dict = torch.load(weight_path)
        model.embed_tokens.load_state_dict({"weight": state_dict.pop("model.embed_tokens.weight")})
        model.norm.load_state_dict({"weight": state_dict.pop("model.norm.weight")})
        model.lm_head.load_state_dict({"weight": state_dict.pop("lm_head.weight")})

        model.eval()
        return model

    def _prepare_forward_data(
        self, seq_input: SeqInput, hidden_states: torch.Tensor, need_serialize: bool
    ) -> Dict[str, Any]:
        if need_serialize:
            hidden_states = serialize_bfloat16_tensor(hidden_states)
        return {"uuid": seq_input.uuid_str_list, "seq_len": seq_input.seq_len_list, "hidden_states": hidden_states}

    def forward(self, inputs_embeds: torch.Tensor, seq_input: SeqInput) -> ForwardResult:
        hidden_states = inputs_embeds
        comm_cost_time_list = []
        last_pp_idx = self.pp_size - 1
        for pp_idx in range(self.pp_size):
            s1 = time.time()
            outputs = self.server.post_sync(
                pp_idx,
                "/forward",
                data=self._prepare_forward_data(seq_input, hidden_states, need_serialize=pp_idx == 0),
            )
            hidden_states = deserialize_bfloat16_tensor(outputs.output) if pp_idx == last_pp_idx else outputs.output
            s2 = time.time()
            comm_cost_time_list.append(s2 - s1 - outputs.cost_time)

        hidden_states = hidden_states.to(self.norm.weight.device)
        # hidden_states: bsz x seq_len x hidden_size
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # logits: bsz x seq_len x vocab_size
        # bsz: 1; seq_len: seq_len1 + seq_len2
        return ForwardResult(logits=logits, comm_cost_time_list=comm_cost_time_list)

    @torch.no_grad()
    async def generate(
        self,
        sequence_request_list: List[SequenceRequestData],
    ) -> AsyncGenerator:
        """
        @params:
            sequence_request_list: List[Params]
                Params:
                    input_ids: torch.Tensor

        """
        uuid_str_list, input_ids_list, seq_len_list = [], [], []
        for sequence_request in sequence_request_list:
            uuid_str_list.append(sequence_request.request_id)
            # 如果是 prefilling，则为 input_ids
            # 否则，为 output_ids[-1]
            # input_ids: bsz x seq_len
            if len(sequence_request.output_ids) == 0:
                input_ids_list.append(sequence_request.input_ids)
                seq_len_list.append(sequence_request.input_ids.shape[-1])
            else:
                input_ids_list.append(torch.tensor([sequence_request.output_ids[-1]]).unsqueeze(0))
                seq_len_list.append(1)

        input_ids = torch.cat(input_ids_list, dim=-1)
        input_embeds = self.embed_tokens(input_ids)

        seq_input = SeqInput(uuid_str_list=uuid_str_list, seq_len_list=seq_len_list)
        forward_result = self(input_embeds, seq_input)
        logits = forward_result.logits

        # 根据 seq 拆开，之后直接在 sampler 中处理
        seq_logits_list = torch.split(logits, seq_input.seq_len_list, dim=1)
        for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
            generate_ids = sequence_request.sampler.decode(seq_logits)
            generate_texts = [self.tok.decode([x]) for x in generate_ids]

            sequence_request.output_ids.append(generate_ids[0])
            sequence_request.generate_ids = generate_ids
            sequence_request.generate_texts = generate_texts

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=self.eos_token_ids,
                max_new_tokens=sequence_request.sampling_params.get("max_new_tokens", 16),
            )
            if end.is_end:
                sequence_request.finish_reason_list = [end.finish_reason]
                sequence_request.is_stop = True
            else:
                sequence_request.output_text += generate_texts[0]  # 不添加 end text

            if len(sequence_request.output_ids) == 1:
                sequence_request.ttft_cost_time = forward_result.comm_cost_time_list
            else:
                sequence_request.tpot_cost_time = forward_result.comm_cost_time_list

        comm_cost_time_list = forward_result.comm_cost_time_list
        self.logger.debug(f"communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")
