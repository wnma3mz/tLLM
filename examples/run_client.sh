#!/bin/bash
# master 的地址，请求分配模型的节点
MASTER_URL=http://mac-mini:8022

# tllm.client --master_addr $MASTER_URL
tllm.client --master_addr $MASTER_URL --is_debug
# tllm.client --master_addr $MASTER_URL --is_debug --config examples/config_one.json --client_idx 0