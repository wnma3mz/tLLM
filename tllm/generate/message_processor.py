from typing import *

from transformers import AutoImageProcessor, AutoProcessor

from tllm.schemas import MIX_TENSOR, SequenceRequestData

from .token_utils import TokenizerUtils

# , processor: Optional[AutoProcessor] = None


def process_mm_input(seq_request: SequenceRequestData, processor: AutoProcessor) -> Dict[str, List[MIX_TENSOR]]:
    if seq_request.multi_modal_inputs is None:
        return {}
    image_processor: AutoImageProcessor = processor.image_processor
    images = seq_request.multi_modal_inputs.get("image", None)
    videos = seq_request.multi_modal_inputs.get("video", None)
    if images:
        image_inputs = image_processor(images=images, videos=None)
        # d["pixel_values"], d["image_grid_thw"]
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = image_processor.merge_size**2
        # 全部放到开头
        seq_request.input_ids = +seq_request.input_ids
        # text[i] = text[i].replace(
        #     "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
        # )
        # index += 1

    if videos:
        d = image_processor(images=None, videos=videos)
        d["pixel_values_videos"], d["video_grid_thw"]
    return {}


class MessageProcessor:
    # TODO async
    def __init__(self, tok: TokenizerUtils):
        self.tok = tok
        self.role_set = {"user", "system", "assistant"}

    # TODO MM Input
    def parse_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        new_messages = []
        for msg in messages:
            assert "role" in msg and "content" in msg, ValueError("role and content must be in message")
            if msg["role"] not in self.role_set:
                raise ValueError(f"role must be in {self.role_set}")
            new_messages.append({"role": msg["role"], "content": msg["content"]})
        return new_messages

    def preprocess(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.tok.preprocess(messages=messages).input_ids

    def fetch_request_id(self, input_ids: List[int]) -> Tuple[Optional[str], int]:
        # max_index, max_id = -1, -1
        # for cache_input_ids, id_ in conversations_dict.items():
        #     index = list_common_prefix(input_ids, cache_input_ids)
        #     if index > max_index:
        #         max_id = id_
        #         max_index = index

        # if max_index == 0 or max_id == -1:
        #     return None, -1
        # return max_id, max_index
        return None, -1
