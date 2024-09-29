import os

import torch
from transformers import LlamaForCausalLM


def split_model_by_layer(model_path: str, save_fname: str, tp: int):
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    state_dict = model.state_dict()
    proj_name_list1 = ["q_proj", "k_proj", "v_proj", "o_proj"]
    proj_name_list2 = ["gate_proj", "up_proj", "down_proj"]
    norm_name_list = ["input_layernorm", "post_attention_layernorm"]
    for i, layer in enumerate(model.model.layers):
        print("Layer", i)
        for tp_idx in range(tp):
            layer_state_dict = {}
            for proj_name in proj_name_list1:
                key_name = f"model.layers.{i}.self_attn.{proj_name}.weight"
                split_dim = 0 if proj_name[0] in "qkv" else 1
                layer_state_dict[key_name] = state_dict[key_name].chunk(tp, dim=split_dim)[tp_idx].clone()
                if tp_idx == tp - 1:
                    state_dict.pop(key_name)
            for proj_name in proj_name_list2:
                key_name = f"model.layers.{i}.mlp.{proj_name}.weight"
                split_dim = 1 if "down" in proj_name else 0
                layer_state_dict[key_name] = state_dict[key_name].chunk(tp, dim=split_dim)[tp_idx].clone()
                if tp_idx == tp - 1:
                    state_dict.pop(key_name)
            for norm_name in norm_name_list:
                key_name = f"model.layers.{i}.{norm_name}.weight"
                layer_state_dict[key_name] = state_dict[key_name]
                if tp_idx == tp - 1:
                    state_dict.pop(key_name)
            torch.save(layer_state_dict, save_fname.format(i, tp_idx))
    torch.save(state_dict, os.path.join(os.path.dirname(save_fname), "other.pth"))


if __name__ == "__main__":
    # model_path = ...
    pipeline_parallel = 1
    save_dir = "./weights/TinyLlama-1.1B-chat-v1.0-pp1"
    # split_model(model_path, save_dir, pipeline_parallel, tensor_parallel=1)
    model_path = "/Users/jianghulu/Documents/TinyLlama-1.1B-Chat-v0.1"
    # model_path = "/Users/lujianghu/Documents/Meta-Llama-3-8B-Instruct"

    base_model_name = os.path.basename(model_path)

    output_dir = os.path.join("weights_tp", base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    tp = 2
    split_model_by_layer(model_path, os.path.join(output_dir, "layer_{}_tp_{}.pth"), tp)
