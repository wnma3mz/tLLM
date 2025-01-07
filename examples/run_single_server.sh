#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=mlx-community/Llama-3.2-1B-Instruct-4bit
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct

tllm.server --model_path $MODEL_PATH --is_local --hostname localhost --client_size 1