import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List

import aiohttp

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_BASE = os.environ.get("TLLM_BENCHMARK_BASE", "http://localhost:8022").rstrip("/")


async def requests_func(messages: List[Dict[str, Any]]):
    url = f"{_BENCHMARK_BASE}/v1/chat/completions"
    data = {
        "messages": messages,
        "model": "tt",
        "stream": False,
        "max_tokens": 200,
    }
    time.sleep(random.random() * 2)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, timeout=100) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return response_data["choices"][0]["message"]["content"], response.status
                else:
                    return None, response.status
    except Exception:
        return None, 500  # 或者返回一个特定的错误码


def print_output(results_async: List[Any]):
    for i, (content, status) in enumerate(results_async):
        print(f"请求 {i+1};", end="")
        if status != 200:
            print(f"  Error: {status}")
        else:
            print(f"  Response: {content}")
        print("-" * 10)


async def main(messages_list: List[List[Dict[str, Any]]], test_type: str):
    print(f"--- {test_type} ---")

    s1 = time.time()
    results_async = await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    print(f"[异步并发请求结果] Time cost (async): {time.time() - s1:.4f} s")

    print_output(results_async)
    print("\n")

    s1 = time.time()
    results_sync = []
    for message in messages_list:
        content, status = await requests_func(message)
        results_sync.append((content, status))
    print(f"[单独请求结果] Time cost (sync): {time.time() - s1:.4f} s")
    print_output(results_sync)
    print("\n")
    return all(status == 200 for _, status in results_async + results_sync)


def load_message():
    path = _REPO_ROOT / "asserts" / "messages.json"
    with open(path, "r") as f:
        messages_dict = json.load(f)
    return messages_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "llm", "vlm"],
        default="all",
        help="Select which benchmark case to run.",
    )
    args = parser.parse_args()

    messages_dict = load_message()
    all_success = True

    if args.mode in {"all", "llm"}:
        all_success = asyncio.run(main(messages_dict["llm"], "纯文本输入测试")) and all_success
    if args.mode in {"all", "vlm"}:
        all_success = asyncio.run(main(messages_dict["vlm"], "图片混合文本输入测试")) and all_success

    if not all_success:
        raise SystemExit(1)
