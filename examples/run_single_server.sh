#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

tllm.server --model_path $MODEL_PATH --is_local --is_debug