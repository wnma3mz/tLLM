import asyncio
import random
import time
from typing import *

from PIL import Image
import aiohttp

from tllm.img_helper import base64_to_pil_image


async def requests_func(prompt: str):
    url = "http://localhost:8022/v1/create_image"
    data = {
        "model": "test",
        "prompt": prompt,
        "config": {
            "num_inference_steps": 2,
            "height": 512,
            "width": 512,
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, timeout=100) as response:
            if response.status == 200:
                response_data = await response.json()
                return response_data
            else:
                print(f"Error: {response.status}")


async def main():
    prompt_list = [
        "a little dog",
        "a little cat",
        "a little mouse",
    ]

    # NOT WORKING
    # messages_list = [messages1, messages2, messages3]
    # print("异步并发请求结果")
    # s1 = time.time()
    # await asyncio.gather(*[requests_func(messages) for messages in messages_list])
    # print(f"time cost: {time.time() - s1:.4f} s")

    print("单独请求结果")
    for i, prompt in enumerate(prompt_list):
        s1 = time.time()
        response_data = await requests_func(prompt)
        if response_data is not None:
            img = Image.open(base64_to_pil_image(response_data["base64"]))
            img.save(f"tt_{i}.png")
        print(f"{i}th image cost time: {time.time() - s1:.4f} s")


if __name__ == "__main__":
    asyncio.run(main())
