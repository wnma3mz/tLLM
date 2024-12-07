import logging
import socket
from typing import Tuple

from tllm import BACKEND, BackendEnum
from tllm.commons.manager import load_master_model
from tllm.engine import AsyncEngine
from tllm.generate import FakeLLMGenerator, LLMGenerator, TokenizerUtils
from tllm.rpc.manager import LocalRPCManager, RPCManager
from tllm.rpc.master_handler import MasterHandler, PendingRequests


def setup_seed(seed: int = 42):
    if BACKEND == BackendEnum.TORCH:
        import torch

        torch.manual_seed(seed)


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)  # 或者其他日志级别

    ch = logging.StreamHandler()
    ch.setLevel(level)  # 控制台输出日志级别

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


async def init_engine(
    logger, model_path: str, master_handler_port: int, is_local: bool, is_fake: bool = False
) -> Tuple[AsyncEngine, TokenizerUtils, MasterHandler]:
    model, tok, num_layers = load_master_model(model_path, logger)
    if is_fake:
        generator = FakeLLMGenerator(None, None, None, None)
        master_handler = None
    elif is_local:
        generator = LLMGenerator(LocalRPCManager(model_path, num_layers), logger, model, tok)
        master_handler = None
    else:
        pending_requests = PendingRequests()
        master_handler = MasterHandler(logger, pending_requests)
        await master_handler.start(master_handler_port)

        generator = LLMGenerator(RPCManager(pending_requests), logger, model, tok)
    engine = AsyncEngine(logger, generator)
    return engine, tok, master_handler
