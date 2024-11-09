#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;

PYTHONPATH="./" python3 -m tllm.entrypoints.api_server --port 8022 --model_path $MODEL_PATH --is_local


