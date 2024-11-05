#!/bin/bash
MASTER_PORT=29501
GRPC_PORT=25001
BASE_PATH=/Users/jianghulu/Documents
# MASTER_URL=ws://192.168.124.24:8000
MASTER_URL=ws://localhost:8022
MODE_SIZE=$1

if [ $MODE_SIZE == "1" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-1B-Instruct-bf16
    START_LAYER_IDX=0
    END_LAYER_IDX=16
elif [ $MODE_SIZE == "3" ]; then
    MODEL_PATH=$BASE_PATH/Llama-3.2-3B-Instruct
    MODEL_PATH=/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-bf16/snapshots/6d88ba43024fef71b10e52e101c7cd4598322601
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

python3 tllm/rpc/client.py --port=$GRPC_PORT --start_layer_idx=$START_LAYER_IDX --end_layer_idx=$END_LAYER_IDX --model_path $MODEL_PATH --master_url $MASTER_URL 