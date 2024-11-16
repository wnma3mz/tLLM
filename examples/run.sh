#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct


export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;

python3 -m tllm.entrypoints.api_server --master_url localhost:25111 --master_handler_port 25111 --port 8022 --model_path $MODEL_PATH --is_debug


