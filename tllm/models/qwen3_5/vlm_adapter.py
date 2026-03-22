from types import MethodType

import mlx.nn as nn
from mlx_vlm.models.qwen3_5.language import LanguageModel as Qwen35VLMLanguageModel
from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as Qwen35MoEVLMLanguageModel


def build_qwen35_language_model_shim(config, is_moe: bool, embed_tokens: nn.Embedding):
    rope_index_impl = (
        Qwen35MoEVLMLanguageModel.get_rope_index if is_moe else Qwen35VLMLanguageModel.get_rope_index
    )

    language_model = type("obj", (object,), {})
    language_model.config = config
    language_model.model = type("obj", (object,), {})
    language_model.model.embed_tokens = embed_tokens
    language_model._position_ids = None
    language_model._rope_deltas = None
    language_model.get_rope_index = MethodType(rope_index_impl, language_model)
    return language_model
