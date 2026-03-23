#!/bin/bash
set -euo pipefail

# 无论从哪执行，都切到仓库根（保证 uv / asserts 路径正确）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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

MODEL_PATH=${MODEL_PATH:-mlx-community/Qwen3.5-9B-4bit}
# MODEL_PATH=${MODEL_PATH:-mlx-community/Qwen3-VL-4B-Instruct-4bit}
TEST_MODE=${TEST_MODE:-llm}
STARTUP_WAIT=${STARTUP_WAIT:-3}
# 轮询次数（每次：先 /health 短超时，再 chat；未就绪时 chat 会快速 500）
READY_TIMEOUT=${READY_TIMEOUT:-120}
HTTP_PORT=${HTTP_PORT:-8022}
# 首次 chat 可能包含 worker 注册后的冷启动推理，2s 过短会导致永远判为未就绪
CHAT_PROBE_TIMEOUT=${CHAT_PROBE_TIMEOUT:-180}
BASE_URL="http://localhost:${HTTP_PORT}"

if [[ "${SKIP_SERVER_START:-0}" == "1" ]]; then
  echo -e "${YELLOW}SKIP_SERVER_START=1：不启动 tllm.server，假定服务已在 ${BASE_URL} 运行${NC}"
  SERVER_PID=""
else
  echo -e "${YELLOW}启动服务...${NC}"
  uv run --no-sync tllm.server --model_path "$MODEL_PATH" --hostname localhost --client_size 1 &
  SERVER_PID=$!
  echo -e "${YELLOW}等待服务启动 (${STARTUP_WAIT}秒)...${NC}"
  sleep "$STARTUP_WAIT"
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${RED}服务启动失败：tllm.server 进程已退出${NC}"
    exit 1
  fi
fi

echo -e "${YELLOW}等待服务就绪（最多轮询 ${READY_TIMEOUT} 次；单次 chat 最长 ${CHAT_PROBE_TIMEOUT}s）...${NC}"
# 勿在循环里反复 uv run python：每次都会冷启动 uv，极慢且易误判为「卡住」
is_ready=0
for ((i=0; i<READY_TIMEOUT; i++)); do
  if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${RED}服务启动失败：tllm.server 进程已退出${NC}"
    exit 1
  fi

  if curl -sf --connect-timeout 3 --max-time 8 "${BASE_URL}/health" >/dev/null 2>&1; then
    code="000"
    code=$(
      curl -sS -o /dev/null -w "%{http_code}" \
        --connect-timeout 5 \
        --max-time "$CHAT_PROBE_TIMEOUT" \
        -H "Content-Type: application/json" \
        -X POST "${BASE_URL}/v1/chat/completions" \
        -d '{"messages":[{"role":"user","content":"ping"}],"model":"tt","stream":false,"max_tokens":1}' || true
    )
    [[ -z "$code" ]] && code="000"
    if [[ "$code" == "200" ]]; then
      is_ready=1
      break
    fi
  fi

  if ((i % 15 == 0 && i > 0)); then
    echo -e "${YELLOW}  …仍在等待就绪（第 ${i}/${READY_TIMEOUT} 次）${NC}"
  fi
  sleep 1
done

if [[ "$is_ready" -ne 1 ]]; then
  echo -e "${RED}服务在 ${READY_TIMEOUT} 次轮询内未就绪（可调大 READY_TIMEOUT / CHAT_PROBE_TIMEOUT）${NC}"
  exit 1
fi

# 运行测试脚本（与上面探测使用同一 BASE_URL）
echo -e "${GREEN}运行测试脚本...${NC}"
TLLM_BENCHMARK_BASE="$BASE_URL" uv run --no-sync python benchmarks/run_async_requests.py --mode "$TEST_MODE"

echo "关闭服务..."
if [[ -n "${SERVER_PID:-}" ]]; then
  kill "$SERVER_PID" 2>/dev/null || true
fi
echo "服务已关闭"
