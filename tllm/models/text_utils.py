from typing import List, Set

from tllm.schemas import GenerateEnd


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


def read_from_text_config(config, attr_name: str):
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        text_config = config
    return getattr(text_config, attr_name)
