#!/bin/bash
GRPC_PORT=25001
# master 的地址，请求分配模型的节点
MASTER_URL=http://m3pro:8022
# 本机地址，用于 master 访问以及其他节点连接
HOSTNAME=m3pro

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

/usr/bin/python3 tllm/rpc/handler.py --port=$GRPC_PORT --master_url $MASTER_URL --ip_addr $HOSTNAME --is_debug
