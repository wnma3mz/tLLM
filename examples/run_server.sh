#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
# MODEL_PATH=/Users/lujianghu/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-bf16/snapshots/f8311090f9ee47782b6f094984a20c856eb841d6
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
MASTER_HOSTNAME=mac-mini

tllm.server --model_path $MODEL_PATH --hostname $MASTER_HOSTNAME --is_debug
# tllm.server --hostname $MASTER_HOSTNAME --model_path $MODEL_PATH --is_debug --config examples/config_one.json