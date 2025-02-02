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
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, timeout=100) as response:
            if response.status == 200:
                response_data = await response.json()
                print(response_data["choices"][0]["message"]["content"])
            else:
                print(f"Error: {response.status}")


async def main(messages_list: List[List[Dict[str, Any]]]):
    # print("异步并发请求结果")
    # s1 = time.time()
    # await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    # print(f"time cost: {time.time() - s1:.4f} s")

    print("单独请求结果")
    s1 = time.time()
    for message in messages_list:
        await requests_func(message)
        print("=" * 20)
    print(f"time cost: {time.time() - s1:.4f} s")


def load_message():
    with open("asserts/debug_messages.json", "r") as f:
        messages_dict = json.load(f)
    return messages_dict


if __name__ == "__main__":
    messages_dict = load_message()

    asyncio.run(main(messages_dict["llm"]))
    # asyncio.run(main(messages_dict["mllm"]))
