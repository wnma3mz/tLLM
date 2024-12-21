import importlib.util

from tllm import BACKEND, BackendEnum
from tllm.models.torch.helper import greedy_decode

MODEL_REGISTER = {}
try:
    # in testing
    from tllm.models.tinygrad.helper import greedy_decode
    from tllm.models.tinygrad.llama import TinyGradLlamaForCausalLM, TinyGradLlamaModel

    MODEL_REGISTER.update({"TinyGradLlamaForCausalLM": (TinyGradLlamaForCausalLM, TinyGradLlamaModel)})
    # MODEL_REGISTER.update({"LlamaForCausalLM": (TinyGradLlamaForCausalLM, TinyGradLlamaModel)})
    # sampling_func = greedy_decode
except ImportError as e:
    pass

if BackendEnum.MLX == BACKEND:
    from tllm.models.mlx.llama import MLXLlamaForCausalLM, MLXLlamaModel
    from tllm.models.mlx.qwen import MLXQwen2ForCausalLM, MLXQwen2Model
    from tllm.models.mlx.qwen_vl.qwen_vl import MLXQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"LlamaForCausalLM": (MLXLlamaForCausalLM, MLXLlamaModel)})
    MODEL_REGISTER.update({"Qwen2ForCausalLM": (MLXQwen2ForCausalLM, MLXQwen2Model)})
    MODEL_REGISTER.update({"Qwen2VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen2Model)})

    if importlib.util.find_spec("mflux"):
        from tllm.models.mlx.flux.flux import Flux1
        from tllm.models.mlx.flux.transformer import FLUXModel

        MODEL_REGISTER.update({"FLUX": (Flux1, FLUXModel)})

    from tllm.models.mlx.helper import greedy_decode

    sampling_func = greedy_decode
elif BackendEnum.TORCH == BACKEND:
    from tllm.models.torch.llama import HFLlamaForCausalLM, HFLlamaModel
    from tllm.models.torch.qwen import HFQwen2ForCausalLM, HFQwen2Model
    from tllm.models.torch.qwen_vl import HFQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"LlamaForCausalLM": (HFLlamaForCausalLM, HFLlamaModel)})
    MODEL_REGISTER.update({"Qwen2ForCausalLM": (HFQwen2ForCausalLM, HFQwen2Model)})
    MODEL_REGISTER.update({"Qwen2VLForConditionalGeneration": (HFQwen2VLForConditionalGeneration, HFQwen2Model)})

    sampling_func = greedy_decode
