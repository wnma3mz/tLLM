import asyncio
import json
import random
import time
from typing import Any, Dict, List

import aiohttp


async def requests_func(messages: List[Dict[str, Any]]):
    url = "http://localhost:8022/v1/chat/completions"
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
    except Exception as e:
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
    results_sync = [await requests_func(message) for message in messages_list]
    print(f"[单独请求结果] Time cost (sync): {time.time() - s1:.4f} s")
    print_output(results_sync)
    print("\n")

    s1 = time.time()
    results_async = await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    print(f"[异步并发请求结果] Time cost (async): {time.time() - s1:.4f} s")
    print_output(results_async)
    print("\n")


def load_message():
    with open("asserts/messages.json", "r") as f:
        messages_dict = json.load(f)
    return messages_dict


if __name__ == "__main__":
    messages_dict = load_message()

    asyncio.run(main(messages_dict["llm"], "纯文本输入测试"))
    asyncio.run(main(messages_dict["mllm"], "图片混合文本输入测试"))
