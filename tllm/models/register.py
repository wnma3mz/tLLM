from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (MyLlamaForCausalLM, MyLlamaModel)}

from tllm import HAS_MLX

if HAS_MLX:
    from tllm.models.mlx_llama import MyMLXLlamaForCausalLM, MyMLXLlamaModel
    from tllm.models.mlx_qwen import MyMLXQwenForCausalLM, MyMLXQwenModel

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (MyMLXLlamaForCausalLM, MyMLXLlamaModel)})
    MODEL_REGISTER.update({"MLXQwen2ForCausalLM": (MyMLXQwenForCausalLM, MyMLXQwenModel)})
