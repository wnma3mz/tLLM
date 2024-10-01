import os

import torch
from transformers import LlamaForCausalLM

if __name__ == "__main__":
    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    model_path = "/Users/lujianghu/Documents/Llama-3.2-3B-Instruct"
    model_path = "/Users/lujianghu/Documents/Meta-Llama-3-8B-Instruct"
    state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()

    save_state_dict = {}
    for key in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]:
        save_state_dict[key] = state_dict.pop(key)

    torch.save(save_state_dict, os.path.join(model_path, "master_weight.pt"))
