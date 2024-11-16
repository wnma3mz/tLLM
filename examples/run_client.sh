#!/bin/bash
MASTER_PORT=29501
GRPC_PORT=25001
MASTER_URL=ws://localhost:8022

MODEL_PATH=/Users/lujianghu/Documents/Llama-3.2-1B-Instruct
START_LAYER_IDX=0
END_LAYER_IDX=16
# 1B 0-16
# 3B 0-28
# 8B 0-32

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --port=$GRPC_PORT --start_layer_idx=$START_LAYER_IDX --end_layer_idx=$END_LAYER_IDX --model_path $MODEL_PATH --master_url $MASTER_URL --is_debug