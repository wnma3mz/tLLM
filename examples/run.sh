#!/bin/bash
BASE_PATH=/Users/lujianghu/Documents/
MODE_SIZE=$1


if [ $MODE_SIZE == "1" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-1B-Instruct
elif [ $MODE_SIZE == "3" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-3B-Instruct
elif [ $MODE_SIZE == "8" ]; then
    MODEL_PATH=$BASE_PATH/Meta-Llama-3-8B-Instruct
elif [ $MODE_SIZE == "70" ]; then
    MODEL_PATH=$BASE_PATH/Meta-Llama-3-70B-Instruct
else 
    echo "Invalid mode size"
    exit 1
fi
WEIGHT_PATH=$MODEL_PATH/master_weight.pt
CONFIG_PATH=./examples/config.json

export PYTHONPATH="./":$PYTHONPATH;
export OMP_NUM_THREADS=8;


python3 -m tllm.entrypoints.server --port 8000 --model_path $MODEL_PATH --weight_path $WEIGHT_PATH --config_path $CONFIG_PATH
# python3 -m tllm.entrypoints.server --port 8000 --model_path $MODEL_PATH --weight_path $WEIGHT_PATH --config_path $CONFIG_PATH --need_start_client


