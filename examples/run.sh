#!/bin/bash
MODEL_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0
WEIGHT_PATH=/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0/master_weight.pt
CONFIG_PATH=./examples/config.json

export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;


# python3 -m tllm.entrypoints.server --port 8000 --model_path $MODEL_PATH --weight_path $WEIGHT_PATH --config_path $CONFIG_PATH
python3 -m tllm.entrypoints.server --port 8000 --model_path $MODEL_PATH --weight_path $WEIGHT_PATH --config_path $CONFIG_PATH --need_start_client


