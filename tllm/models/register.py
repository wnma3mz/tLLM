from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (MyLlamaForCausalLM, MyLlamaModel)}

from tllm import HAS_MLX

if HAS_MLX:
    from tllm.models.mlx_llama import MyMLXLlamaForCausalLM, MyMLXLlamaModel

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (MyMLXLlamaForCausalLM, MyMLXLlamaModel)})
