#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=mlx-community/Llama-3.2-1B-Instruct-8bit
# MODEL_PATH=mlx-community/Llama-3.2-3B-Instruct-bf16
# MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-bf16
# MODEL_PATH=/Users/lujianghu/.cache/huggingface/hub/models--kaetemi--Meta-Llama-3.1-8B-Q4_0-GGUF/snapshots/5209d67d7c69db2fd6edad402596fcb4e6ece939/meta-llama-3.1-8b-q4_0.gguf
# MODEL_PATH=/Users/lujianghu/.cache/huggingface/hub/models--happylife39--Llama-3.2-1B-Q4_0-GGUF/snapshots/d0d40e5341984332d03bfab0f9afca23647585c5/llama-3.2-1b-q4_0.gguf
MASTER_HANDLER_PORT=25111
MASTER_URL=localhost:$MASTER_HANDLER_PORT
HTTP_PORT=8022

export PYTHONPATH="./":$PYTHONPATH;

/usr/bin/python3 -m tllm.entrypoints.api_server --master_url $MASTER_URL --master_handler_port $MASTER_HANDLER_PORT --port $HTTP_PORT --model_path $MODEL_PATH --is_local --is_debug


