#!/bin/bash
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
MODEL_PATH=mlx-community/Qwen2.5-VL-3B-Instruct-4bit 
# MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=mlx-community/Qwen2.5-0.5B-Instruct-bf16
# MODEL_PATH=~/Documents/models--Qwen2.5-0.5B-Instruct-4bit
# MODEL_PATH=wnma3mz/DeepSeek-R1-Distill-Qwen-7B-4bit
# MODEL_PATH=~/Documents/DeepSeek-R1-Distill-Qwen-7B-4bit

tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1