#!/bin/bash
MODEL_PATH=mlx-community/Qwen3.5-0.8B-4bit

uv run --no-sync tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1
# tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1 --is_debug
