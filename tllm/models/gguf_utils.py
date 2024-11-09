# Copyright © 2023 Apple Inc.
# Modified by https://github.com/ml-explore/mlx-examples/blob/main/llms/gguf_llm/models.py

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from huggingface_hub import snapshot_download
import mlx.core as mx


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    context_length: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


@dataclass
class GGUFConfig:
    context_length: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool
    model_type: str = None
    decoder_start_layer_idx: int = 0
    decoder_end_layer_idx: int = 0
    is_merge: bool = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def get_config(metadata: dict) -> GGUFConfig:
    output = {
        "context_length": metadata["llama.context_length"],
        "hidden_size": metadata["llama.embedding_length"],
        "num_hidden_layers": int(metadata["llama.block_count"]),
        "num_attention_heads": metadata["llama.attention.head_count"],
        "intermediate_size": metadata["llama.feed_forward_length"],
        "num_key_value_heads": metadata["llama.attention.head_count_kv"],
        "rms_norm_eps": metadata["llama.attention.layer_norm_rms_epsilon"],
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "rope_theta": metadata["llama.rope.freq_base"],
        "rope_traditional": True,
    }
    return GGUFConfig(**output)


def translate_weight_names(name):
    name = name.replace("blk.", "model.layers.")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("token_embd", "model.embed_tokens")
    name = name.replace("output_norm", "model.norm")
    name = name.replace("output", "lm_head")
    return name


def load_gguf_weight(gguf_file: str, repo: str = None):
    # If the gguf_file exists, try to load model from it.
    # Otherwise try to download and cache from the HF repo
    if not Path(gguf_file).exists():
        if repo is None:
            raise ValueError(f"Could not find file {gguf_file}, and no Hugging Face" " repo provided for download.")
        model_path = snapshot_download(
            repo_id=repo,
            allow_patterns=[gguf_file],
        )
        if not (Path(model_path) / gguf_file).exists():
            raise ValueError(f"File {gguf_file} not in repo {repo}.")
        gguf_file = str(Path(model_path) / gguf_file)

    print(f"[INFO] Loading model from {gguf_file}")
    weights, metadata = mx.load(gguf_file, return_metadata=True)
    gguf_ft = metadata["general.file_type"]
    if gguf_ft == 0 or gguf_ft == 1:
        # ALL_F32 or MOSTLY_F16
        quantization = None
        pass
    elif gguf_ft == 2 or gguf_ft == 3:
        # MOSTLY_Q4_0 or MOSTLY_Q4_1
        quantization = {"group_size": 32, "bits": 4}
        # print bits value
        print(f"{quantization['bits']} bits quantized model")
    elif gguf_ft == 7:
        # MOSTLY_Q8_0 = 7
        quantization = {"group_size": 32, "bits": 8}
        print(f"{quantization['bits']} bits quantized model")
    else:
        quantization = None
        print("[WARNING] Using unsupported GGUF quantization. Casting to float16.")

    weights = {translate_weight_names(k): v for k, v in weights.items()}
    config = get_config(metadata)
    config.quantization = quantization
    return weights, config, metadata
