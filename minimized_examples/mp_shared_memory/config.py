import os
from pathlib import Path


class Config:
    # 项目基础配置
    BASE_DIR = Path(__file__).parent
    LOG_DIR = BASE_DIR / "logs"

    # 共享内存配置
    BUFFER_SIZE = 1024 * 1024  # 1MB
    REQUEST_BUFFER_NAME = "engine_buffer"
    RESPONSE_BUFFER_NAME = "response_buffer"

    # API配置
    API_HOST = "127.0.0.1"
    API_PORT = 8000

    # 进程配置
    ENGINE_PROCESS_COUNT = 1  # 引擎进程数量

    @classmethod
    def init(cls):
        # 创建日志目录
        os.makedirs(cls.LOG_DIR, exist_ok=True)
