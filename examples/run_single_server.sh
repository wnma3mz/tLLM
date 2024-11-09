#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-8bit/snapshots/d48cdf0a4ea22d893b7c63a99d6a693e24822795
# MODEL_PATH=/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3-8B-Instruct-4bit/snapshots/c38b3b1f03cce0ce0ccd235e5c97b0d3d255e651
export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;

PYTHONPATH="./" python3 -m tllm.entrypoints.api_server --port 8022 --model_path $MODEL_PATH --is_local


