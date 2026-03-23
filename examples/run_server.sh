#!/bin/bash
MODEL_PATH=mlx-community/Qwen3.5-0.8B-4bit
MASTER_HOSTNAME=mac-mini

uv run tllm.server --model_path $MODEL_PATH --hostname $MASTER_HOSTNAME --client_size 2
# tllm.server --hostname $MASTER_HOSTNAME --model_path $MODEL_PATH --is_debug --config examples/config_one.json