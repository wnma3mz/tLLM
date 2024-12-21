#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
MODEL_PATH=/Users/lujianghu/Documents/flux/schnell_4bit
MASTER_HOSTNAME=mac-mini

export PYTHONPATH="./":$PYTHONPATH;

python3 -m tllm.entrypoints.api_server --hostname $MASTER_HOSTNAME --model_path $MODEL_PATH --client_size 1 --is_debug