#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/mflux/schnell_4bit
MASTER_HANDLER_PORT=25111
MASTER_URL=localhost:$MASTER_HANDLER_PORT
HTTP_PORT=8022

export PYTHONPATH="./":$PYTHONPATH;
python3 -m tllm.entrypoints.image_api_server --master_url $MASTER_URL --master_handler_port $MASTER_HANDLER_PORT --port $HTTP_PORT --model_path $MODEL_PATH --is_local --is_debug


