import logging
import logging.handlers
import os
from pathlib import Path
from typing import Literal, Optional

level = logging.INFO


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

        if cls.logger:
            cls._add_file_handler(cls.logger)

    @classmethod
    def _add_file_handler(cls, logger: logging.Logger):
        if cls._log_file:
            log_dir = os.path.dirname(cls._log_file)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)

            # 移除现有的文件处理器（如果有）
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    logger.removeHandler(handler)

            # 添加新的文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                cls._log_file, maxBytes=cls._max_bytes, backupCount=cls._backup_count, encoding="utf-8"
            )
            file_handler.setLevel(cls._level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    @classmethod
    def _setup_logger(cls, name: Literal["master", "handler"]) -> logging.Logger:
        if cls.logger is None:  # 仅在第一次调用时创建logger
            cls.logger = logging.getLogger(name)
            cls.logger.setLevel(cls._level)

            if not cls.logger.hasHandlers():  # 检查是否已经存在 handlers
                ch = logging.StreamHandler()
                ch.setLevel(cls._level)
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                ch.setFormatter(formatter)
                cls.logger.addHandler(ch)

            cls._add_file_handler(cls.logger)  # 始终添加文件handler
        return cls.logger

    @classmethod
    def setup_master_logger(cls) -> logging.Logger:
        cls.set_log_file("master.log")
        return cls._setup_logger("master")

    @classmethod
    def setup_handler_logger(cls, name: Optional[str]) -> logging.Logger:
        cls.set_log_file("handler.log")
        return cls._setup_logger("handler" if name is None else name)
