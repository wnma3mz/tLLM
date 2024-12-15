#!/bin/bash
# master 的地址，请求分配模型的节点
MASTER_URL=http://mac-mini:8022

export OMP_NUM_THREADS=8;
export PYTHONPATH="./":$PYTHONPATH;

python3 -m tllm.entrypoints.handler.handler --master_addr $MASTER_URL --is_debug