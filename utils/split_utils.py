import os

import numpy as np
import torch
from transformers import LlamaForCausalLM


def split_model(model_path: str, save_dir: str, pipeline_parallel: int, tensor_parallel: int):
    # 按照 tensor_parallel, pipeline_parallel 切分模型
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict()["model.embed_tokens"], os.path.join(save_dir, "embed_tokens.pth"))
    torch.save(model.state_dict()["lm_head"], os.path.join(save_dir, "lm_head.pth"))
    torch.save(model.state_dict()["model.norm"], os.path.join(save_dir, "norm.pth"))

    assert len(model.model.layers) % pipeline_parallel == 0
    step = len(model.model.layers) // pipeline_parallel
    for i in range(pipeline_parallel):
        params_state_dict = model.model.layers[i * step : (i + 1) * step].state_dict()
        torch.save({k: v.clone() for k, v in params_state_dict.items()}, os.path.join(save_dir, f"decoder_pp{i}.pth"))


def split_model_by_layer(model_path: str, save_fname: str):
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(torch.bfloat16)

    state_dict = model.state_dict()
    proj_name_list1 = ["q_proj", "k_proj", "v_proj", "o_proj"]
    proj_name_list2 = ["gate_proj", "up_proj", "down_proj"]
    norm_name_list = ["input_layernorm", "post_attention_layernorm"]
    for i, layer in enumerate(model.model.layers):
        print("Layer", i)
        layer_state_dict = {}
        for proj_name in proj_name_list1:
            key_name = f"model.layers.{i}.self_attn.{proj_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name]
            state_dict.pop(key_name)
        for proj_name in proj_name_list2:
            key_name = f"model.layers.{i}.mlp.{proj_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name]
            state_dict.pop(key_name)
        for norm_name in norm_name_list:
            key_name = f"model.layers.{i}.{norm_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name]
            state_dict.pop(key_name)
        torch.save(layer_state_dict, save_fname.format(i))
    torch.save(state_dict, os.path.join(os.path.dirname(save_fname), "other.pth"))


def split_model_by_layer_numpy(model_path: str, save_fname: str):
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(torch.bfloat16)
    target_dtype = torch.float16 # for convert to numpy

    state_dict = model.state_dict()
    proj_name_list1 = ["q_proj", "k_proj", "v_proj", "o_proj"]
    proj_name_list2 = ["gate_proj", "up_proj", "down_proj"]
    norm_name_list = ["input_layernorm", "post_attention_layernorm"]
    for i, layer in enumerate(model.model.layers):
        print("Layer", i)
        layer_state_dict = {}
        for proj_name in proj_name_list1:
            key_name = f"model.layers.{i}.self_attn.{proj_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name].to(target_dtype).numpy()
            state_dict.pop(key_name)
        for proj_name in proj_name_list2:
            key_name = f"model.layers.{i}.mlp.{proj_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name].to(target_dtype).numpy()
            state_dict.pop(key_name)
        for norm_name in norm_name_list:
            key_name = f"model.layers.{i}.{norm_name}.weight"
            layer_state_dict[key_name] = state_dict[key_name].to(target_dtype).numpy()
            state_dict.pop(key_name)
        np.save(save_fname.format(i), layer_state_dict)
    state_dict = {k: v.to(target_dtype).numpy() for k, v in state_dict.items()}
    np.save(os.path.join(os.path.dirname(save_fname), "other.npy"), state_dict)


if __name__ == "__main__":
    # model_path = ...
    pipeline_parallel = 1
    save_dir = "./weights/TinyLlama-1.1B-chat-v1.0-pp1"
    # split_model(model_path, save_dir, pipeline_parallel, tensor_parallel=1)

    # model_path = "/Users/lujianghu/Documents/Meta-Llama-3-8B-Instruct"
    # os.makedirs("./weights/llama3-8b", exist_ok=True)
    # split_model_by_layer(model_path, "./weights/llama3-8b/layer_{}.pth")

    # model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    # os.makedirs("./np_weights/tiny-llama", exist_ok=True)
    # split_model_by_layer_numpy(model_path, "./np_weights/tiny-llama/layer_{}.npy")

    model_path = "/Users/lujianghu/Documents/Meta-Llama-3-8B-Instruct"
    os.makedirs("./np_weights/llama3-8b", exist_ok=True)
    split_model_by_layer_numpy(model_path, "./np_weights/llama3-8b/layer_{}.npy")
