#!/bin/bash
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct 
# MODEL_PATH=mlx-community/Qwen2.5-VL-3B-Instruct-4bit 

tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1