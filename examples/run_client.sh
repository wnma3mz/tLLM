#!/bin/bash
GRPC_PORT=25001
MASTER_URL=http://localhost:8022

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --is_debug