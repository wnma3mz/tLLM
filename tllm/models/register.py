from tllm.models.llama import TLlamaForCausalLM, LlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (TLlamaForCausalLM, LlamaModel)}

from tllm import HAS_MLX

if HAS_MLX:
    from tllm.models.mlx_llama import MLXLlamaForCausalLM, MLXLlamaModel
    from tllm.models.mlx_qwen import MLXQwen2ForCausalLM, MLXQwen2Model
    from tllm.models.mlx_qwen_vl import MLXQwen2VLForConditionalGeneration

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (MLXLlamaForCausalLM, MLXLlamaModel)})
    MODEL_REGISTER.update({"MLXQwen2ForCausalLM": (MLXQwen2ForCausalLM, MLXQwen2Model)})
    MODEL_REGISTER.update({"MLXQwen2VLForConditionalGeneration": (MLXQwen2VLForConditionalGeneration, MLXQwen2Model)})
