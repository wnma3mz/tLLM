#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
MODEL_PATH=/Users/lujianghu/Documents/flux/schnell_4bit
MASTER_HOSTNAME=mac-mini

export PYTHONPATH="./":$PYTHONPATH;

python3 -m tllm.entrypoints.image_api_server --ip_addr $MASTER_HOSTNAME --model_path $MODEL_PATH --is_debug