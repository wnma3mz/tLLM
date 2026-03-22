#!/bin/bash
set -euo pipefail

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo -e "${YELLOW}启动服务...${NC}"

# 启动服务并在后台运行
MODEL_PATH=${MODEL_PATH:-mlx-community/Qwen3.5-0.8B-4bit}
# MODEL_PATH=${MODEL_PATH:-mlx-community/Qwen3-VL-4B-Instruct-4bit}
TEST_MODE=${TEST_MODE:-llm}
STARTUP_WAIT=${STARTUP_WAIT:-3}
READY_TIMEOUT=${READY_TIMEOUT:-240}

uv run --no-sync tllm.server --model_path $MODEL_PATH --hostname localhost --client_size 1 &

# 保存服务进程ID
SERVER_PID=$!

# 等待服务启动
echo -e "${YELLOW}等待服务启动 (${STARTUP_WAIT}秒)...${NC}"
sleep "$STARTUP_WAIT"

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo -e "${RED}服务启动失败：tllm.server 进程已退出${NC}"
  exit 1
fi

echo -e "${YELLOW}等待服务就绪（最长 ${READY_TIMEOUT} 秒）...${NC}"
is_ready=0
for ((i=0; i<READY_TIMEOUT; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${RED}服务启动失败：tllm.server 进程已退出${NC}"
    exit 1
  fi

  if uv run --no-sync python - <<'PY'
import sys
import requests

data = {
    "messages": [{"role": "user", "content": "ping"}],
    "model": "tt",
    "stream": False,
    "max_tokens": 1,
}

try:
    resp = requests.post("http://localhost:8022/v1/chat/completions", json=data, timeout=2)
    sys.exit(0 if resp.status_code == 200 else 1)
except Exception:
    sys.exit(1)
PY
  then
    is_ready=1
    break
  fi

  sleep 1
done

if [[ "$is_ready" -ne 1 ]]; then
  echo -e "${RED}服务在 ${READY_TIMEOUT} 秒内未就绪${NC}"
  exit 1
fi

# 运行测试脚本
echo -e "${GREEN}运行测试脚本...${NC}"
uv run --no-sync python benchmarks/run_async_requests.py --mode "$TEST_MODE"

echo "关闭服务..."
kill "$SERVER_PID" 2>/dev/null || true
echo "服务已关闭"