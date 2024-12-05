import time
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoProcessor

from tllm.models.register import sampling_func
from tllm.models.utils import is_generate_end
from tllm.schemas import MIX_TENSOR, ForwardResult, SeqInput, SequenceRequestData


def merge_mm_input(mm_input_list: List[Dict[str, List[np.ndarray]]]) -> Optional[Dict[str, List[MIX_TENSOR]]]:
    if all([x is None for x in mm_input_list]) or all([len(x) == 0 for x in mm_input_list]):
        return None
    pixel_values_list, image_grid_thw_list, pixel_values_videos_list, video_grid_thw_list = [], [], [], []
    for x in mm_input_list:
        if "image" in x:
            pixel_values_list.append(x["image"]["pixel_values"])
            image_grid_thw_list.append(x["image"]["image_grid_thw"])
        if "video" in x:
            pixel_values_videos_list.append(x["video"]["pixel_values_videos"])
            video_grid_thw_list.append(x["video"]["video_grid_thw"])

    pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else None
    pixel_values_videos = np.concatenate(pixel_values_videos_list, axis=0) if pixel_values_videos_list else None
    image_grid_thw = np.concatenate(image_grid_thw_list, axis=0) if image_grid_thw_list else None
    video_grid_thw = np.concatenate(video_grid_thw_list, axis=0) if video_grid_thw_list else None

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
    }


def process_mm_input(
    seq_request: SequenceRequestData, processor: AutoProcessor, **kwargs
) -> Dict[str, List[np.ndarray]]:
    if seq_request.multi_modal_inputs is None and len(seq_request.multi_modal_inputs) == 0:
        return {}
    assert processor is not None
    image_processor: AutoImageProcessor = processor.image_processor
    images = seq_request.multi_modal_inputs.get("image", None)
    videos = seq_request.multi_modal_inputs.get("video", None)

    vision_start_id = kwargs["vision_start_id"]
    vision_end_id = kwargs["vision_end_id"]

    response_dict = {}
    if images:
        image_inputs = image_processor(images=images, videos=None)
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = image_processor.merge_size**2
        # 全部放到开头
        image_token_id = kwargs["image_token_id"]
        image_input_ids = []
        for x in image_grid_thw:
            repeat_times = x.prod() // merge_length
            image_input_ids += [vision_start_id] + [image_token_id] * repeat_times + [vision_end_id]
        seq_request.input_ids = image_input_ids + seq_request.input_ids
        response_dict.update({"image": image_inputs})
    if videos:
        video_inputs = image_processor(images=None, videos=videos)
        video_grid_thw = video_inputs["image_grid_thw"]
        merge_length = image_processor.merge_size**2
        # 全部放到开头
        video_end_id = kwargs["video_end_id"]
        image_input_ids = []
        for x in video_grid_thw:
            repeat_times = x.prod() // merge_length
            image_input_ids += [vision_start_id] + [video_end_id] * repeat_times + [vision_end_id]
        seq_request.input_ids = image_input_ids + seq_request.input_ids
        response_dict.update({"video": video_inputs})
    return response_dict


class LLMGenerator:
    def __init__(self, manager: "RPCManager", logger, model, tok: "TokenizerUtils") -> None:
        self.manager = manager
        self.logger = logger
        self.model = model
        self.tok = tok
        self.processor = getattr(model, "processor", None)
        self.mm_config = getattr(model, "mm_config", None)

    def update_url(self, url: str, pp_size: int):
        self.manager.update_url(url, pp_size)

    async def forward(self, inputs_embeds: MIX_TENSOR, seq_input: SeqInput) -> ForwardResult:
        s1 = time.perf_counter()
        hidden_states, calc_cost_time_list = await self.manager.forward(inputs_embeds, seq_input)
        comm_cost_time = time.perf_counter() - s1 - sum(calc_cost_time_list)
        return ForwardResult(
            hidden_states=hidden_states,
            comm_cost_time=comm_cost_time,
            calc_cost_time=sum(calc_cost_time_list),
        )

    @torch.inference_mode()
    async def generate(self, sequence_request_list: List[SequenceRequestData]) -> AsyncGenerator:
        """
        @params:
            sequence_request_list: List[SequenceRequestData]
                Params:
                    input_ids: torch.Tensor

        """
        uuid_list, input_ids_list, seq_len_list, mm_input_list = [], [], [], []
        for sequence_request in sequence_request_list:
            uuid_list.append(sequence_request.request_id)
            # 如果是 prefilling，则为 input_ids; 否则，为 output_ids[-1]
            # input_ids: seq_len
            if sequence_request.is_prefill:
                # if sequence_request.history_request_id:
                #     uuid_list[-1] = sequence_request.history_request_id
                if self.processor is not None:
                    mm_input_list.append(process_mm_input(sequence_request, self.processor, **self.mm_config))
                input_ids_list.append(np.array(sequence_request.input_ids))
                # seq_len_list.append(sequence_request.q_len) # 需要搭配 history_request_id 使用
                seq_len_list.append(len(sequence_request.input_ids))
            else:
                input_ids_list.append(np.array([sequence_request.output_ids[-1]]))
                seq_len_list.append(1)

        mm_input = merge_mm_input(mm_input_list)
        input_ids = np.concatenate(input_ids_list, axis=-1)
        # [seq_len1 + seq_len2 + ...] -> [seq_len1 + seq_len2 + ..., hidden_size]
        if mm_input is None:
            input_embeds = self.model.get_input_embeddings(input_ids)
        else:
            input_embeds = self.model.get_input_embeddings(input_ids, **mm_input)

        seq_input = SeqInput(uuid_list=uuid_list, seq_len_list=seq_len_list)
        s0 = time.perf_counter()
        forward_result = await self.forward(input_embeds, seq_input)
        self.logger.debug(f"decoder cost time: {time.perf_counter() - s0:.4f}s")
        s1 = time.perf_counter()
        seq_logits: List[MIX_TENSOR] = self.model.get_logits(forward_result.hidden_states)
        self.logger.debug(f"logits cost time: {time.perf_counter() - s1:.4f}s")
        assert seq_logits.shape[0] == len(sequence_request_list)

        s1 = time.perf_counter()
        # TODO: batch decode by group
        # TODO: sequence_request.sampling_params
        seq_generate_ids: List[int] = sampling_func(seq_logits)
        seq_generate_texts = self.tok.decode(seq_generate_ids)

        for generate_id, generate_text, sequence_request in zip(
            seq_generate_ids, seq_generate_texts, sequence_request_list
        ):
            sequence_request.output_ids.append(generate_id)

            end = is_generate_end(
                sequence_request.output_ids,
                eos_token_ids=sequence_request.sampling_params.stop_token_ids,
                max_tokens=sequence_request.sampling_params.max_tokens,
            )
            if end.is_end:
                sequence_request.finish_reason_list = [end.finish_reason]
                sequence_request.is_stop = True
            else:
                sequence_request.generate_text = generate_text
                sequence_request.output_text += generate_text  # 不添加 end text
            print("generate text: ", generate_text)

            if sequence_request.is_prefill:
                sequence_request.ttft_cost_time = time.perf_counter() - s0
                sequence_request.decode_start_ts = time.perf_counter()
                sequence_request.is_prefill = False

        fraction = forward_result.comm_cost_time / (forward_result.comm_cost_time + forward_result.calc_cost_time)
        self.logger.debug(f"de tokenizer cost time: {time.perf_counter() - s1:.4f}s")
        self.logger.debug(f"communication cost time: {forward_result.comm_cost_time:.4f}s({fraction*100:.1f}%)")
        self.logger.debug("=" * 5)


class FakeLLMGenerator(LLMGenerator):

    async def generate(self, sequence_request_list: List[SequenceRequestData]) -> AsyncGenerator:
        generate_id, generate_text = 0, "fake "
        for sequence_request in sequence_request_list:
            sequence_request.output_ids.append(generate_id)
            if len(sequence_request.output_ids) == sequence_request.sampling_params.max_tokens:
                sequence_request.finish_reason_list = ["length"]
                sequence_request.is_stop = True
            else:
                sequence_request.generate_text = generate_text
                sequence_request.output_text += generate_text
