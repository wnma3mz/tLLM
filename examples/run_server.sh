#!/bin/bash
# MODEL_PATH=mlx-community/Meta-Llama-3.1-8B-Instruct-bf16
# MODEL_PATH=/Users/lujianghu/.cache/huggingface/hub/models--mlx-community--Meta-Llama-3.1-8B-Instruct-bf16/snapshots/f8311090f9ee47782b6f094984a20c856eb841d6
MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
MASTER_HOSTNAME=mac-mini
GRPC_PORT=25111
HTTP_PORT=8022                             # 本地 HTTP 服务的端口，websocket 前端页面

MASTER_URL=$MASTER_HOSTNAME:$GRPC_PORT # master 的地址，用于其他客户端连接

export PYTHONPATH="./":$PYTHONPATH;
# num_hidden_layers
# 1B  16
# 3B  28
# 8B  32
# 70B 70

/usr/bin/python3 -m tllm.entrypoints.api_server --master_url $MASTER_URL --master_handler_port $GRPC_PORT --port $HTTP_PORT --model_path $MODEL_PATH --is_debug