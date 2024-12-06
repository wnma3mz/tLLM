from tllm import HAS_MLX
from tllm.models.torch.helper import greedy_decode
from tllm.models.torch.llama import HFLlamaForCausalLM, HFLlamaModel

sampling_func = greedy_decode
MODEL_REGISTER = {"LlamaForCausalLM": (HFLlamaForCausalLM, HFLlamaModel)}


try:
    # in testing
    from tllm.models.tinygrad.helper import greedy_decode
    from tllm.models.tinygrad.llama import TinyGradLlamaForCausalLM, TinyGradLlamaModel

    # MODEL_REGISTER.update({"TinyGradLlamaForCausalLM": (TinyGradLlamaForCausalLM, TinyGradLlamaModel)})
    MODEL_REGISTER.update({"LlamaForCausalLM": (TinyGradLlamaForCausalLM, TinyGradLlamaModel)})
    sampling_func = greedy_decode
except ImportError as e:
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
