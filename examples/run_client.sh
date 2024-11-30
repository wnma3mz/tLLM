#!/bin/bash
GRPC_PORT=25001
MASTER_URL=http://192.168.124.24:8022 # master 地址，用于连接
IP_ADDR="192.168.124.5" # 本机地址，用于 master 访问

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $IP_ADDR --is_debug