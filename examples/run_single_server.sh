#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
# MODEL_PATH=/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e
MODEL_PATH=/Users/jianghulu/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4
export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;

PYTHONPATH="./" python3 -m tllm.entrypoints.api_server --master_url localhost:25111 --master_handler_port 25111 --port 8022 --model_path $MODEL_PATH --is_local --is_debug


