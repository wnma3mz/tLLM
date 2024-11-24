#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;

MASTER_HANDLER_PORT=25111
MASTER_URL=localhost:$MASTER_HANDLER_PORT
# 1B 0-16
# 3B 0-28
# 8B 0-32

python3 -m tllm.entrypoints.api_server --master_url $MASTER_URL --master_handler_port $MASTER_HANDLER_PORT --port 8022 --model_path $MODEL_PATH --is_debug


