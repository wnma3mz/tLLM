import json
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoConfig

from tllm.schemas import GenerateEnd


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
