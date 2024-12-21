#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/flux/schnell_4bit
MASTER_HOSTNAME=mac-mini

export PYTHONPATH="./":$PYTHONPATH;
python3 -m tllm.entrypoints.api_server --ip_addr $MASTER_HOSTNAME --model_path $MODEL_PATH --is_local --is_debug --is_image


