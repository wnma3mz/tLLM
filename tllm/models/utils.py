import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoConfig

from tllm import HAS_MLX
from tllm.schemas import MIX_TENSOR, GenerateEnd

if HAS_MLX:
    import mlx.core as mx

    cat_func = lambda tensors: mx.concat(tensors, axis=0)
else:
    import torch

    cat_func = lambda tensors: torch.cat(tensors, dim=0)


def is_generate_end(output_ids: List[int], eos_token_ids: Set[int], max_tokens: int) -> GenerateEnd:
    if len(output_ids) >= max_tokens:
        return GenerateEnd(finish_reason="length", is_end=True)

    if output_ids[-1] in eos_token_ids:
        return GenerateEnd(finish_reason="stop", is_end=True)

    return GenerateEnd(finish_reason=None, is_end=False)


def get_weight_path(model_path: str) -> Tuple[set, List[str]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    file_set = set()
    prefix_key_list = ["model.embed_tokens.", "model.norm.", "lm_head."]
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        for key, file_ in index["weight_map"].items():
            for prefix_key in prefix_key_list:
                if key.startswith(prefix_key):
                    file_set.add(file_)
    else:
        file_set.add("model.safetensors")
    return file_set, prefix_key_list


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)
    # TODO: setting timeout
    if not model_path.exists():
        print("Model not found locally. Downloading from Hugging Face Hub.")
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except:
            raise ValueError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            )
    return model_path


def read_eos_token_ids(config: AutoConfig) -> Set[int]:
    eos_token_ids = set()
    if hasattr(config, "eos_token_ids"):
        if isinstance(config.eos_token_id, list):
            eos_token_ids |= set(config.eos_token_ids)
        else:
            eos_token_ids.add(config.eos_token_id)
    return eos_token_ids


def pop_weight_func(
    prefix_key_list: List[str], weights: Dict[str, MIX_TENSOR], num_layers: int, start_idx: int, end_idx: int
) -> Dict[str, MIX_TENSOR]:
    prefix_key_list += [f"model.layers.{i}." for i in range(num_layers) if not (start_idx <= i < end_idx)]
    key_list = list(weights.keys())
    for key in key_list:
        for prefix_key in prefix_key_list:
            if key.startswith(prefix_key):
                weights.pop(key)
    return weights


def merge_weight_func(
    layer_pattern: re.Pattern, name_list: str, cat_name_fmt: str, weights: Dict[str, MIX_TENSOR]
) -> Dict[str, MIX_TENSOR]:
    key_list = list(weights.keys())

    temp_w = {}  # save merge weights
    merge_num = len(name_list)
    for key in key_list:
        res = layer_pattern.findall(key)
        if res:
            layer_idx = int(res[0])
            if layer_idx not in temp_w:
                temp_w[layer_idx] = {}
        else:
            continue

        for name in name_list:
            if name in key:
                temp_w[layer_idx].update({name: weights.pop(key)})

        layer_weights = temp_w.get(layer_idx, [])
        if len(layer_weights) == merge_num:
            name = cat_name_fmt.format(layer_idx=layer_idx)
            weights[name] = cat_func([layer_weights[qkv] for qkv in name_list])
            temp_w.pop(layer_idx)

    layer_idx_list = list(temp_w.keys())
    for layer_idx in layer_idx_list:
        if len(temp_w[layer_idx]) != 0:
            raise ValueError(
                f"merge [{cat_name_fmt}] failed, layer_idx: {layer_idx}, temp_w: {temp_w[layer_idx].keys()}"
            )
    return weights


def default_merge_attn_bias(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.bias")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.bias"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)


def default_merge_mlp_weight(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    mlp_pattern = re.compile(r"model\.layers\.(\d+)\.mlp.*.weight")
    mlp_list = ["gate_proj", "up_proj"]
    mlp_name = "model.layers.{layer_idx}.mlp.gate_up_proj.layer.weight"
    return merge_weight_func(mlp_pattern, mlp_list, mlp_name, weights)


def default_merge_attn_weight(weights: Dict[str, MIX_TENSOR]) -> Dict[str, MIX_TENSOR]:
    attn_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn.*.weight")
    attn_list = ["q_proj", "k_proj", "v_proj"]
    attn_name = "model.layers.{layer_idx}.self_attn.qkv_proj.layer.weight"
    return merge_weight_func(attn_pattern, attn_list, attn_name, weights)
