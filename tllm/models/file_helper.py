import os
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import constants, snapshot_download
from huggingface_hub.file_download import repo_folder_name


def get_hf_cache_model_path(repo_id: str, revision: Optional[str] = None) -> Path:
    cache_dir = constants.HF_HUB_CACHE
    if revision is None:
        revision = constants.DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    repo_type = "model"
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))

    ref_path = os.path.join(storage_folder, "refs", revision)
    commit_hash = None
    if os.path.exists(ref_path):
        # retrieve commit_hash from refs file
        with open(ref_path) as f:
            commit_hash = f.read()

    # Try to locate snapshot folder for this commit hash
    if commit_hash is not None:
        snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
        if os.path.exists(snapshot_folder):
            # Snapshot folder exists => let's return it
            # (but we can't check if all the files are actually there)
            return snapshot_folder
    return None


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = get_hf_cache_model_path(path_or_hf_repo, revision)
        if model_path is None:
            print("Model not found locally. Downloading from Hugging Face Hub.")
            try:
                # TODO: setting timeout
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


def parse_model_size(model_name: str) -> float:
    part_list = model_name.lower().split("-")[::-1]
    model_size = -1
    for part in part_list:
        if part.endswith("b"):
            try:
                model_size = float(part[:-1])
                break
            except:
                pass
    assert model_size > 0, f"Invalid model name: {model_name}"
    return model_size


def auto_set_client_size(model_size: float) -> int:
    # num_hidden_layers
    # 1B  16
    # 3B  28
    # 8B  32
    # 70B 70
    if model_size < 4:
        return 1
    elif model_size <= 8:
        return 2
    elif model_size <= 32:
        return 4
    elif model_size <= 72:
        return 8
    else:
        raise ValueError(f"Model size {model_size} is too large")


def split_model_layers(total_layers: int, client_size: int) -> List[Tuple[int, int]]:
    # 根据 model size 和 层数来划分客户端数量以及每个客户端的层数
    each_client_layers = total_layers // client_size
    return [(start_idx, start_idx + each_client_layers) for start_idx in range(0, total_layers, each_client_layers)]
