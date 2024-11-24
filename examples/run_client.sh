#!/bin/bash
GRPC_PORT=25001
MASTER_URL=http://localhost:8022
IP_ADDR="localhost"

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $IP_ADDR --is_debug