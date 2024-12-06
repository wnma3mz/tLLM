import asyncio
import random
import time
from typing import *

import aiohttp


async def requests_func(messages: List[Dict[str, Any]]):
    url = "http://localhost:8022/v1/chat/completions"
    data = {
        "messages": messages,
        "model": "tt",
        # "stream": True
        "stream": False,
        "max_tokens": 20,
    }
    time.sleep(random.random() * 2)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, timeout=100) as response:
            if response.status == 200:
                response_data = await response.json()
                print(response_data["choices"][0]["message"]["content"])
            else:
                print(f"Error: {response.status}")


async def main():
    messages1 = [{"role": "user", "content": "Hello, how are you?"}]
    messages2 = [{"role": "user", "content": "Hello, What's your name?"}]
    messages3 = [{"role": "user", "content": "今天天气怎么样"}]

    messages_list = [messages1, messages2, messages3]
    print("异步并发请求结果")
    s1 = time.time()
    await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    print(f"time cost: {time.time() - s1:.4f} s")

    print("单独请求结果")
    s1 = time.time()
    await requests_func(messages3)
    await requests_func(messages1)
    await requests_func(messages2)
    print(f"time cost: {time.time() - s1:.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
