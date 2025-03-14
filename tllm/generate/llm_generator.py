import copy
import time
from typing import Callable, List

import numpy as np

from tllm.generate.token_utils import TokenizerUtils
from tllm.grpc.master_service.worker_manager import WorkerRPCManager
from tllm.models.register import sampling_func
from tllm.models.utils import is_generate_end
from tllm.schemas import MIX_TENSOR, ForwardResult, SeqInput, SequenceRequestData
from tllm.singleton_logger import SingletonLogger


class LLMGenerator:
    def __init__(self, manager: WorkerRPCManager, model) -> None:
        self.manager = manager
        self.logger = SingletonLogger.setup_master_logger()
        self.model = model
        self.tok: TokenizerUtils = model.tok
        self.merge_mm_input: Callable = getattr(model, "merge_mm_input", None)

    async def forward(self, inputs_embeds: MIX_TENSOR, seq_input: SeqInput) -> ForwardResult:
        s1 = time.perf_counter()
        hidden_states, calc_cost_time_list = await self.manager.forward(inputs_embeds, seq_input)
        comm_cost_time = time.perf_counter() - s1 - sum(calc_cost_time_list)
        return ForwardResult(
            hidden_states=hidden_states,
            comm_cost_time=comm_cost_time,
            calc_cost_time=sum(calc_cost_time_list),
        )

    async def process_input(self, request_list: List[SequenceRequestData]):
        uuid_list, input_ids_list, mm_input_list = [], [], []
        for sequence_request in request_list:
            uuid_list.append(sequence_request.request_id)
            # 如果是 prefilling，则为 input_ids; 否则，为 output_ids[-1]
            if sequence_request.is_prefill:
                if sequence_request.multi_modal_inputs is not None:
                    mm_input_list.append(sequence_request.multi_modal_inputs)

                input_ids_list.append(np.array(sequence_request.input_ids))

                if sequence_request.is_gen_image:
                    compare_input_ids = copy.deepcopy(input_ids_list[-1])
                    uuid_list.append(sequence_request.request_id + "-bak")  # For Janus-Pro, 每个 request 对应两个句子
                    compare_input_ids[1:-1] = self.model.pad_token_id
                    input_ids_list.append(compare_input_ids)
            else:
                input_ids_list.append(np.array([sequence_request.output_ids[-1]]))
                if sequence_request.is_gen_image:
                    uuid_list.append(sequence_request.request_id + "-bak")
                    input_ids_list.append(np.array([sequence_request.output_ids[-1]]))

        mm_input = self.merge_mm_input(mm_input_list) if self.merge_mm_input is not None else None

        # [seq_len1 + seq_len2 + ...] -> [seq_len1 + seq_len2 + ..., hidden_size]
        input_ids = np.concatenate(input_ids_list, axis=-1)

        seq_input = SeqInput(uuid_list=uuid_list, input_ids_list=input_ids_list)
        return seq_input, input_ids, mm_input

    async def process_output(self, request_list: List[SequenceRequestData], seq_logits: MIX_TENSOR, s0: float):
        # TODO: batch decode by group
        # TODO: sequence_request.sampling_params
        seq_generate_ids: List[int] = sampling_func(seq_logits)
        seq_generate_texts, cache_token_ids_list = self.tok.decode(
            seq_generate_ids, [x.cache_token_ids for x in request_list]
        )

        for seq_idx, sequence_request in enumerate(request_list):
            generate_id = seq_generate_ids[seq_idx]
            generate_id = generate_id[-1] if isinstance(generate_id, list) else generate_id
            generate_text = seq_generate_texts[seq_idx]
            sequence_request.cache_token_ids = cache_token_ids_list[seq_idx]

            sequence_request.output_ids.append(generate_id)

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=sequence_request.sampling_params.stop_token_ids,
                max_tokens=sequence_request.sampling_params.max_tokens,
            )
            if end.is_end:
                # len(sequence_request.output_ids) == 576
                if sequence_request.is_gen_image:
                    generate_text = self.model.decode_image(sequence_request.output_ids)
                    sequence_request.generate_text = generate_text
                    sequence_request.output_text = generate_text

                sequence_request.finish_reason_list = [end.finish_reason]
                sequence_request.is_stop = True
            else:
                # eos时，不添加 end text
                sequence_request.generate_text = generate_text
                sequence_request.output_text += generate_text

            if sequence_request.is_prefill:
                sequence_request.ttft_cost_time = time.perf_counter() - s0
                sequence_request.decode_start_ts = time.perf_counter()
                sequence_request.is_prefill = False

    async def generate(self, request_list: List[SequenceRequestData]):
        """
        @params:
            request_list: List[SequenceRequestData]
                Params:
                    input_ids: List[int]

        """
        is_gen_image = any(x.is_gen_image for x in request_list)  # In Experiment
        seq_input, input_ids, mm_input = await self.process_input(request_list)

        if mm_input is not None:
            input_embeds = self.model.get_input_embeddings(input_ids, **mm_input)
        else:
            if not is_gen_image:
                input_embeds = self.model.get_input_embeddings(input_ids)
            else:
                if request_list[0].is_prefill:
                    input_embeds = self.model.get_input_embeddings(input_ids)
                else:
                    # For Janus-Pro, 在生成图片时，生成第二个 token 后
                    input_embeds = self.model.get_gen_img_embeds(input_ids)

        s0 = time.perf_counter()
        forward_result = await self.forward(input_embeds, seq_input)
        self.logger.debug(f"decoder cost time: {time.perf_counter() - s0:.4f}s")
        s1 = time.perf_counter()
        if is_gen_image:
            # For Janus-Pro
            seq_logits: List[MIX_TENSOR] = self.model.get_gen_head(forward_result.hidden_states)
        else:
            seq_logits: List[MIX_TENSOR] = self.model.get_logits(forward_result.hidden_states)

        self.logger.debug(f"logits cost time: {time.perf_counter() - s1:.4f}s")
        assert seq_logits.shape[0] == len(request_list)

        s1 = time.perf_counter()
        await self.process_output(request_list, seq_logits, s0)

        fraction = forward_result.comm_cost_time / (forward_result.comm_cost_time + forward_result.calc_cost_time)
        self.logger.debug(f"de tokenizer cost time: {time.perf_counter() - s1:.4f}s")
        self.logger.debug(f"communication cost time: {forward_result.comm_cost_time:.4f}s({fraction*100:.1f}%)")
        self.logger.debug("=" * 5)
