import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import *

import requests


class Server:
    def __init__(self, url_list):
        self.url_list = url_list
        self.executor = ThreadPoolExecutor()

    def __len__(self):
        return len(self.url_list)

    # 异步 post
    def post(self, path, data_list: List[Dict[str, Any]]):
        response_list = []
        for url, data in zip(self.url_list, data_list):
            response = requests.post(f"{url}{path}", json=data)
            response_list.append(response)
        return response_list

    # 指定 api path，对所有 url 发送 post 请求
    def post_thread(self, path, data_list: List[Dict[str, Any]]) -> List[requests.Response]:
        response_list = []
        futures = []

        for url, data in zip(self.url_list, data_list):
            future = self.executor.submit(requests.post, f"{url}{path}", json=data)
            futures.append(future)

        for future in futures:
            response_list.append(future.result())
        return response_list

    # 指定 url_idx，多线程请求
    def post_thread_url(self, url_idx, path, data_list: List[Dict[str, Any]]) -> List[requests.Response]:
        response_list = []
        futures = []
        url = self.url_list[url_idx]
        for data in data_list:
            future = self.executor.submit(requests.post, f"{url}{path}", json=data)
            futures.append(future)

        for future in futures:
            response_list.append(future.result())
        return response_list

    # 指定 api path, url_idx 以及每个 url 的请求 data_list 多线程请求
    def post_thread_url_dict(
        self, path: str, url_idx_data_dict: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[requests.Response]]:
        # 返回结果按照 dict 顺序
        response_dict = {}
        futures = []

        for url_idx, data_list in url_idx_data_dict.items():
            for data in data_list:
                future = self.executor.submit(requests.post, f"{self.url_list[url_idx]}{path}", json=data)
                futures.append(future)

        for url_idx, data_list in url_idx_data_dict.items():
            response_dict[url_idx] = []
            for _ in data_list:
                response_dict[url_idx].append(futures.pop(0).result())
        return response_dict

    # 单个 post
    def post_sync(self, url_idx: int, path, data):
        return self._post(self.url_list[url_idx], path, data)

    def _post(self, url, path, data):
        return requests.post(f"{url}{path}", json=data)

    def _get(self, url, path):
        return requests.get(f"{url}{path}")

    def fetch_list_output(self, response_list: Union[List[requests.Response], requests.Response]) -> List:
        if isinstance(response_list, list):
            return [response.json()["output"] for response in response_list]
        return response_list.json()["output"]

    def is_success(self, response_list: Union[List[requests.Response], requests.Response]) -> bool:
        if isinstance(response_list, list):
            return all(response.status_code == 200 for response in response_list)
        return response_list.status_code == 200
