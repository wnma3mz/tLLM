#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=mlx-community/Llama-3.2-1B-Instruct-8bit
# MODEL_PATH=mlx-community/Llama-3.2-3B-Instruct-bf16
# MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-bf16
MASTER_HANDLER_PORT=25111
MASTER_URL=localhost:$MASTER_HANDLER_PORT
HTTP_PORT=8022

export PYTHONPATH="./":$PYTHONPATH;

/usr/bin/python3 -m tllm.entrypoints.api_server --master_url localhost:25111 --master_handler_port $MASTER_HANDLER_PORT --port $HTTP_PORT --model_path $MODEL_PATH --is_local --is_debug


