#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
# MODEL_PATH=/Users/lujianghu/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-bf16/snapshots/f8311090f9ee47782b6f094984a20c856eb841d6
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
MASTER_HOSTNAME=mac-mini

export PYTHONPATH="./":$PYTHONPATH;
# num_hidden_layers
# 1B  16
# 3B  28
# 8B  32
# 70B 70

python3 -m tllm.entrypoints.api_server --ip_addr $MASTER_HOSTNAME --model_path $MODEL_PATH --is_debug