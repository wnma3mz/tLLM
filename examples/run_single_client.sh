#!/bin/bash
MASTER_PORT=29501
GRPC_PORT=25001
BASE_PATH=/Users/lujianghu/Documents/
MASTER_URL=ws://localhost:8022
MODE_SIZE=$1
TP=$2


if [ $MODE_SIZE == "1" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-1B-Instruct
    START_LAYER_IDX=0
    END_LAYER_IDX=16
elif [ $MODE_SIZE == "3" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-3B-Instruct
    START_LAYER_IDX=0
    END_LAYER_IDX=28
elif [ $MODE_SIZE == "8" ]; then
    MODEL_PATH=$BASE_PATH/Meta-Llama-3-8B-Instruct
    START_LAYER_IDX=0
    END_LAYER_IDX=32
elif [ $MODE_SIZE == "70" ]; then
    MODEL_PATH=$BASE_PATH/Meta-Llama-3-70B-Instruct
else 
    echo "Invalid mode size"
    exit 1
fi

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

torchrun --nproc_per_node=$TP --master_port=$MASTER_PORT tllm/rpc/client.py --port=$GRPC_PORT --start_layer_idx=$START_LAYER_IDX --end_layer_idx=$END_LAYER_IDX --model_path $MODEL_PATH --master_url $MASTER_URL 