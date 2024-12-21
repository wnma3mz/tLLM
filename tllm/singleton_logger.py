import logging
from typing import Literal, Optional

level = logging.INFO


class SingletonLogger:
    logger = None
    _level = logging.INFO  # 默认日志级别

    @classmethod
    def set_level(cls, level_name=Optional[str]):
        """设置日志级别并更新现有logger"""
        # 将字符串日志级别转换为logging常量
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level_map.get(level_name.upper(), logging.INFO)
        cls._level = level

        # 如果logger已经存在，更新其级别
        if cls.logger:
            cls.logger.setLevel(level)
            for handler in cls.logger.handlers:
                handler.setLevel(level)

    @classmethod
    def _setup_logger(cls, name=Literal["master", "handler"]) -> logging.Logger:
        """创建基础logger的内部方法"""
        if cls.logger is None:
            cls.logger = logging.getLogger(name)
            cls.logger.setLevel(cls._level)

            ch = logging.StreamHandler()
            ch.setLevel(cls._level)

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)

            cls.logger.addHandler(ch)
        return cls.logger

    @classmethod
    def setup_master_logger(cls) -> logging.Logger:
        return cls._setup_logger("master")

    @classmethod
    def setup_handler_logger(cls, name: Optional[str]) -> logging.Logger:
        return cls._setup_logger("handler" if name is None else name)
