from transformers import LlamaForCausalLM

from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyLlamaModel)}

try:
    import mlx.core as mx  # type: ignore

    HAS_MLX = True
except:
    HAS_MLX = False

if HAS_MLX:
    from tllm.models.mlx_llama import MyMLXLlamaModel

    MODEL_REGISTER.update({{"MLXLlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyMLXLlamaModel)}})
