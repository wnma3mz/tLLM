import time
from typing import *

class CacheManager:
    # 管理每个 client 的 past_key_values，即 kv_cache
    # max_alive_time: 超过多久没有访问就删除，单位秒
    def __init__(self, max_alive_time=60):
        self.max_alive_time = max_alive_time
        self.cache_dict = {}

    def get(self, key) -> Any:
        return self.cache_dict.get(key)

    def set(self, key, value: Any) -> None:
        self.cache_dict[key] = {"past_key_values": value, "ts": time.time()}

    def delete(self, key):
        self.cache_dict.pop(key)

    def clear(self):
        self.cache_dict.clear()

    def check_alive(self):
        now = time.time()
        key_list = list(self.cache_dict.keys())
        for key in key_list:
            if now - self.cache_dict[key]["ts"] > self.max_alive_time:
                self.cache_dict.pop(key)
