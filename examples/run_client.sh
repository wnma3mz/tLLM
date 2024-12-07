#!/bin/bash
GRPC_PORT=25001
# master 的地址，请求分配模型的节点
MASTER_URL=http://192.168.124.27:8022
# 本机地址，用于 master 访问以及其他节点连接
IP_ADDR="192.168.124.27"

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

/usr/bin/python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $IP_ADDR --is_debug