from typing import Dict, List, Optional, Set, Union

import numpy as np

from tllm.schemas import MIX_TENSOR, GenerateEnd


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


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


def default_process_mm_input(
    multi_modal_inputs: Dict[str, Union[List, str]], image_processor
) -> Dict[str, List[np.ndarray]]:
    multi_modal_dict = {}
    multi_modal_dict["text"] = multi_modal_inputs["text"]

    images = multi_modal_inputs.get("image", None)
    videos = multi_modal_inputs.get("video", None)

    if images:
        image_inputs = image_processor(images=images, videos=None)
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = image_processor.merge_size**2
        # 全部放到开头
        image_input_text = ""
        for x in image_grid_thw:
            repeat_times = x.prod() // merge_length
            image_input_text += "<|vision_start|>" + "<|image_pad|>" * repeat_times + "<|vision_end|>"
        multi_modal_dict["text"] = image_input_text + multi_modal_dict["text"]
        multi_modal_dict.update({"image": image_inputs})
    if videos:
        video_inputs = image_processor(images=None, videos=videos)
        video_grid_thw = video_inputs["image_grid_thw"]
        merge_length = image_processor.merge_size**2
        image_input_text = ""
        for x in video_grid_thw:
            repeat_times = x.prod() // merge_length
            image_input_text += "<|vision_start|>" + "<|video_pad|>" * repeat_times + "<|vision_end|>"
        multi_modal_dict["text"] = image_input_text + multi_modal_dict["text"]
        multi_modal_dict.update({"video": video_inputs})
    return multi_modal_dict


def read_from_text_config(config, attr_name: str):
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        text_config = config
    return getattr(text_config, attr_name)
