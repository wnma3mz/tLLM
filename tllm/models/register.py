# coding: utf-8
import importlib.util

from tllm import BACKEND, BackendEnum
from tllm.models.torch.helper import greedy_decode

MODEL_REGISTER = {}
DEP_MODEL_REGISTER = {}


if BackendEnum.MLX == BACKEND:
    from tllm.models.mlx.janus_pro.janus_pro import MLXJanusProConditionalGeneration
    from tllm.models.mlx.llama import MLXLlamaForCausalLM, MLXLlamaModel
    from tllm.models.mlx.qwen2 import MLXQwen2ForCausalLM, MLXQwen2Model
    from tllm.models.mlx.qwen3 import MLXQwen3ForCausalLM, MLXQwen3Model

    MODEL_REGISTER.update({"LlamaForCausalLM": (MLXLlamaForCausalLM, MLXLlamaModel)})
    MODEL_REGISTER.update({"Qwen2ForCausalLM": (MLXQwen2ForCausalLM, MLXQwen2Model)})
    MODEL_REGISTER.update({"Qwen3ForCausalLM": (MLXQwen3ForCausalLM, MLXQwen3Model)})
    MODEL_REGISTER.update({"Qwen3MoeForCausalLM": (MLXQwen3ForCausalLM, MLXQwen3Model)})
    MODEL_REGISTER.update({"JanusProConditionalGeneration": (MLXJanusProConditionalGeneration, MLXLlamaModel)})

    from tllm.models.mlx.qwen2_vl import MLXQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"Qwen2VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen2Model)})
    MODEL_REGISTER.update({"Qwen2_5_VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen2Model)})
    MODEL_REGISTER.update({"Qwen3VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen3Model)})
    MODEL_REGISTER.update({"Qwen3VLMoeForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen3Model)})

    from tllm.models.mlx.helper import greedy_decode

    sampling_func = greedy_decode
elif BackendEnum.TORCH == BACKEND:
    from tllm.models.torch.llama import HFLlamaForCausalLM, HFLlamaModel
    from tllm.models.torch.qwen2 import HFQwen2ForCausalLM, HFQwen2Model
    from tllm.models.torch.qwen2_vl import HFQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"LlamaForCausalLM": (HFLlamaForCausalLM, HFLlamaModel)})
    MODEL_REGISTER.update({"Qwen2ForCausalLM": (HFQwen2ForCausalLM, HFQwen2Model)})
    MODEL_REGISTER.update({"Qwen2VLForConditionalGeneration": (HFQwen2VLForConditionalGeneration, HFQwen2Model)})

    sampling_func = greedy_decode
