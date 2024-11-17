import asyncio
import random
import time
from typing import *

import aiohttp


async def requests_func(messages: List[Dict[str, Any]]):
    # url = "http://192.168.1.4:8022/v1/chat/completions"
    url = "http://localhost:8022/v1/chat/completions"

    data = {
        "messages": messages,
        "model": "tt",
        # "stream": True
        "stream": False,
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
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里面有什么？"},
                {"type": "image_url", "image_url": {"file_path": "asserts/image-2.png"}},
            ],
        }
    ]
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image_url", "image_url": {"file_path": "asserts/image-1.png"}},
            ],
        }
    ]
    print("单独请求结果")
    s1 = time.time()
    await requests_func(messages1)
    print(f"time cost: {time.time() - s1:.4f} s")

    messages_list = [messages1, messages2]
    print("异步并发请求结果")
    s1 = time.time()
    await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    print(f"time cost: {time.time() - s1:.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
