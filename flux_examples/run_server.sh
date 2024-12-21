#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
MODEL_PATH=/Users/lujianghu/Documents/flux/schnell_4bit
MASTER_HOSTNAME=mac-mini

tllm.server --model_path $MODEL_PATH --hostname $MASTER_HOSTNAME --client_size 1 --is_debug