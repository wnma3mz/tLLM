#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
# MODEL_PATH=mlx-community/Llama-3.2-1B-Instruct-4bit
MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
MASTER_HOSTNAME=mac-mini

tllm.server --model_path $MODEL_PATH --hostname $MASTER_HOSTNAME --is_debug --client_size 1
# tllm.server --hostname $MASTER_HOSTNAME --model_path $MODEL_PATH --is_debug --config examples/config_one.json