from typing import List, Optional

from tllm.schemas import MIX_TENSOR, SamplingParams


class SamplerUtils:
    """Legacy sampling utility kept for compatibility.

    The current inference path uses backend-specific sampler functions registered in `tllm.models.register`.
    """

    def __init__(self, method: str) -> None:
        self.method = method
        assert self.method in ["greedy", "beam_search", "sampling"]

    def sampling(self, logits: MIX_TENSOR, sampling_params: Optional[SamplingParams] = None) -> List[int]:
        raise NotImplementedError(
            "SamplerUtils is deprecated in MLX-only mode. Use sampling function from model register instead."
        )
