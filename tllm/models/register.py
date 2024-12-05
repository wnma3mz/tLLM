from tllm.models.torch.llama import LlamaModel, TLlamaForCausalLM

MODEL_REGISTER = {"LlamaForCausalLM": (TLlamaForCausalLM, LlamaModel)}

from tllm import HAS_MLX

try:
    from tllm.models.tinygrad.llama import TinyGradLlamaForCausalLM, TinyGradLlamaModel

    MODEL_REGISTER.update({"TinyGradLlamaForCausalLM": (TinyGradLlamaForCausalLM, TinyGradLlamaModel)})
except ImportError:
    pass

if HAS_MLX:
    from tllm.models.mlx.llama import MLXLlamaForCausalLM, MLXLlamaModel
    from tllm.models.mlx.qwen import MLXQwen2ForCausalLM, MLXQwen2Model
    from tllm.models.mlx.qwen_vl import MLXQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"LlamaForCausalLM": (MLXLlamaForCausalLM, MLXLlamaModel)})
    MODEL_REGISTER.update({"Qwen2ForCausalLM": (MLXQwen2ForCausalLM, MLXQwen2Model)})
    MODEL_REGISTER.update({"Qwen2VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen2Model)})

    from tllm.models.mlx.helper import greedy_decode

    sampling_func = greedy_decode
else:
    from tllm.models.torch.helper import greedy_decode

    sampling_func = greedy_decode

    # from tinygrad.tensor import Tensor
    # from typing import List
    # def sampling_func(logits: Tensor) -> List[int]:
    #     return logits.argmax(axis=-1).tolist()
