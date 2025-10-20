#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}启动服务...${NC}"

# 启动服务并在后台运行
MODEL_PATH=mlx-community/Qwen3-VL-4B-Instruct-3bit
tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1 --is_image &

# 保存服务进程ID
SERVER_PID=$!

# 等待服务启动
echo -e "${YELLOW}等待服务启动 (10秒)...${NC}"
sleep 3

# 运行测试脚本
echo -e "${GREEN}运行测试脚本...${NC}"
python3 benchmarks/run_async_requests.py

echo "关闭服务..."
kill $SERVER_PID
echo "服务已关闭"