#!/bin/bash
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-bf16
MODEL_PATH=~/Documents/models--Qwen2.5-0.5B-Instruct-4bit

tllm.server --model_path $MODEL_PATH --is_local --hostname localhost --client_size 1