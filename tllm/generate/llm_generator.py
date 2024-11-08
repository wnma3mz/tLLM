import itertools
import time
from typing import AsyncGenerator, List, Union

import numpy as np
import torch

from tllm import HAS_MLX
from tllm.models.protocol import ForwardResult, SeqInput, SequenceRequestData
from tllm.models.utils import is_generate_end
from tllm.rpc.manager import RPCManager

if HAS_MLX:
    import mlx.core as mx


class LLMGenerator:
    def __init__(self, server: RPCManager, logger, model):
        self.server = server
        self.pp_size = len(server)
        self.logger = logger
        self.model = model

    def forward(self, inputs_embeds, seq_input: SeqInput) -> ForwardResult:
        hidden_states = inputs_embeds
        comm_cost_time_list, calc_cost_time_list = [], []
        for pp_idx in range(self.pp_size):
            is_first = pp_idx == 0
            is_last = pp_idx == self.pp_size - 1
            s1 = time.perf_counter()
            hidden_states, pp_cost_time = self.server.forward(
                pp_idx, hidden_states, seq_input, is_first, is_last, to_tensor=not HAS_MLX
            )
            comm_cost_time_list.append(time.perf_counter() - s1 - pp_cost_time)
            calc_cost_time_list.append(pp_cost_time)
        return ForwardResult(
            hidden_states=hidden_states,
            comm_cost_time_list=comm_cost_time_list,
            calc_cost_time_list=calc_cost_time_list,
        )

    @torch.no_grad()
    async def generate(self, sequence_request_list: List[SequenceRequestData]) -> AsyncGenerator:
        """
        @params:
            sequence_request_list: List[SequenceRequestData]
                Params:
                    input_ids: torch.Tensor

        """
        uuid_list, input_ids_list, seq_len_list = [], [], []
        for sequence_request in sequence_request_list:
            uuid_list.append(sequence_request.request_id)
            # 如果是 prefilling，则为 input_ids; 否则，为 output_ids[-1]
            # input_ids: bsz x seq_len
            if sequence_request.is_prefill:
                if sequence_request.history_request_id:
                    uuid_list[-1] = sequence_request.history_request_id
                input_ids_list.append(np.array(sequence_request.input_ids))
                seq_len_list.append(sequence_request.q_len)
            else:
                input_ids_list.append(np.array([sequence_request.output_ids[-1]]))
                seq_len_list.append(1)

        input_ids = np.concatenate(input_ids_list, axis=-1)
        # [seq_len1 + seq_len2 + ...] -> [seq_len1 + seq_len2 + ..., hidden_size]
        input_embeds = self.model.get_input_embeddings(input_ids)

        seq_input = SeqInput(uuid_list=uuid_list, seq_len_list=seq_len_list)
        s0 = time.perf_counter()
        forward_result = self.forward(input_embeds, seq_input)
        self.logger.debug(f"decoder cost time: {time.perf_counter() - s0:.4f}s")
        s1 = time.perf_counter()
        seq_logits_list: List[Union[torch.Tensor, "mx.array"]] = self.model.get_logits(
            forward_result.hidden_states, seq_len_list
        )
        assert len(seq_logits_list) == len(sequence_request_list)

        s1 = time.perf_counter()
        # 根据 seq 拆开，之后直接在 sampler 中处理
        for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
            generate_ids = sequence_request.sampler.sampling(seq_logits, sequence_request.sampling_params)
            generate_texts = sequence_request.sampler.decode(generate_ids)
            sequence_request.output_ids.append(generate_ids[0])

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=self.model.eos_token_ids,
                max_tokens=sequence_request.sampling_params.max_tokens,
            )
            if end.is_end:
                sequence_request.finish_reason_list = [end.finish_reason]
                sequence_request.is_stop = True
            else:
                sequence_request.generate_text = generate_texts[0]
                sequence_request.output_text += generate_texts[0]  # 不添加 end text

            if sequence_request.is_prefill:
                sequence_request.ttft_cost_time = time.perf_counter() - s0
                sequence_request.decode_start_ts = time.perf_counter()
                sequence_request.is_prefill = False

        comm_cost_time_list = forward_result.comm_cost_time_list
        comm_cost_time_str = ",".join([f"{x:.4f}" for x in comm_cost_time_list])
        sum_comm = sum(comm_cost_time_list)
        fraction = sum_comm / (sum_comm + sum(forward_result.calc_cost_time_list))
        self.logger.debug(f"de tokenizer cost time: {time.perf_counter() - s1:.4f}s")
        self.logger.debug(f"communication cost time: {comm_cost_time_str}s({fraction*100:.1f}%)")
        self.logger.debug("=" * 5)
