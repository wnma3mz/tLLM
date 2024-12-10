#!/bin/bash
# master 的地址，请求分配模型的节点
MASTER_URL=http://mac-mini:8022
# 本机地址，用于 master 访问以及其他节点连接
HOSTNAME=mac-mini

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 tllm/rpc/handler.py --master_addr $MASTER_URL --is_debug
