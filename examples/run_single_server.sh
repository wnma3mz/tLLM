#!/bin/bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct 
MODEL_PATH=mlx-community/Qwen3-0.6B-4bit
# MODEL_PATH=mlx-community/Qwen3-30B-A3B-4bit
# MODEL_PATH=mlx-community/Qwen2.5-VL-3B-Instruct-4bit 

tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1 # --is_debug