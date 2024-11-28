#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
MASTER_HANDLER_PORT=25111
MASTER_URL=localhost:$MASTER_HANDLER_PORT
HTTP_PORT=8022

export PYTHONPATH="./":$PYTHONPATH;
# 1B 0-16
# 3B 0-28
# 8B 0-32

python3 -m tllm.entrypoints.api_server --master_url $MASTER_URL --master_handler_port $MASTER_HANDLER_PORT --port $HTTP_PORT --model_path $MODEL_PATH --is_debug


