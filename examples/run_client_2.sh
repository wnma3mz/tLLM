#!/bin/bash
GRPC_PORT=25002
MASTER_URL=http://192.168.124.27:8022
IP_ADDR="192.168.124.27"

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

/usr/bin/python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $IP_ADDR --is_debug