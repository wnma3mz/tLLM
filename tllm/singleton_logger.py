import logging
import logging.config
import os
from pathlib import Path
from typing import Literal, Optional

level = logging.INFO

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s,%(msecs)03d - %(levelprefix)s %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",  # 添加日期格式
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            # 在这里修改 fmt 字符串，添加 %(asctime)s 和 %(msecs)03d
            # 并设置 datefmt
            "fmt": '%(asctime)s,%(msecs)03d - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",  # 设置日期时间的基本格式
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            # 你可以将访问日志输出到 stdout 或 stderr
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "master": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "handler": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


class SingletonLogger:
    logger = None
    _level = logging.INFO  # 默认日志级别
    _log_file = None  # 日志文件路径
    _max_bytes = 10 * 1024 * 1024  # 默认每个日志文件最大10MB
    _backup_count = 5  # 默认保留5个备份文件

    @classmethod
    def set_level(cls, level_name=Optional[str]):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level_map.get(level_name.upper(), logging.INFO)
        cls._level = level

        if cls.logger:
            cls.logger.setLevel(level)
            for handler in cls.logger.handlers:
                handler.setLevel(level)

    @classmethod
    def set_log_file(cls, log_file: str, max_bytes: int = None, backup_count: int = None):
        cls._log_file = log_file
        if max_bytes is not None:
            cls._max_bytes = max_bytes
        if backup_count is not None:
            cls._backup_count = backup_count

    @classmethod
    def _setup_logger(cls, name: Literal["master", "handler"]) -> logging.Logger:
        if cls.logger is None:  # 仅在第一次调用时创建logger
            logging.config.dictConfig(LOGGING_CONFIG)
            cls.logger = logging.getLogger(name)
            cls.logger.setLevel(cls._level)
        return cls.logger

    @classmethod
    def setup_master_logger(cls) -> logging.Logger:
        cls.set_log_file("master.log")
        return cls._setup_logger("master")

    @classmethod
    def setup_handler_logger(cls, name: Optional[str]) -> logging.Logger:
        cls.set_log_file("handler.log")
        return cls._setup_logger("handler" if name is None else name)
