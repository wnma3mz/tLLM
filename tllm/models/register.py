from transformers import LlamaForCausalLM

from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyLlamaModel)}

from tllm import HAS_MLX

if HAS_MLX:
    from tllm.models.mlx_llama import MyMLXLlamaForCausalLM, MyMLXLlamaModel

    MODEL_REGISTER.update({"MLXLlamaForCausalLM": (LlamaForCausalLM, MyMLXLlamaForCausalLM, MyMLXLlamaModel)})
