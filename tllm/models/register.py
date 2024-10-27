from transformers import LlamaForCausalLM

from tllm.models.llama import MyLlamaForCausalLM, MyLlamaModel

MODEL_REGISTER = {"LlamaForCausalLM": (LlamaForCausalLM, MyLlamaForCausalLM, MyLlamaModel)}
