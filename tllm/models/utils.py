from typing import List, Set

from transformers import AutoConfig

from tllm.schemas import GenerateEnd


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


def read_eos_token_ids(config: AutoConfig) -> Set[int]:
    eos_token_ids = set()
    if hasattr(config, "eos_token_ids"):
        if isinstance(config.eos_token_id, list):
            eos_token_ids |= set(config.eos_token_ids)
        else:
            eos_token_ids.add(config.eos_token_id)
    return eos_token_ids
