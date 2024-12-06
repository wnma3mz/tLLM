#!/bin/bash
GRPC_PORT=25001
# master 地址，用于连接
MASTER_URL=http://localhost:8022
# 本机地址，用于 master 访问
IP_ADDR="localhost"

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $IP_ADDR --is_debug