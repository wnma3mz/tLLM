import asyncio
import concurrent.futures

import requests

# 并发发送处理请求
data_list = [
    {"message": "test1", "data": 123},
    {"message": "test2", "data": 456},
]

url = "http://localhost:8000/process"
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(requests.post, url, json=data) for data in data_list]

for future in concurrent.futures.as_completed(futures):
    response = future.result()
    print(response.json())
