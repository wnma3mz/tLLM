import argparse
import logging
import socket
from typing import *

import torch

from tllm.engine import AsyncEngine
from tllm.generate import LLMGenerator, TokenizerUtils
from tllm.models.manager import load_master_model
from tllm.rpc.manager import LocalRPCManager, RPCManager
from tllm.rpc.master_handler import MasterHandler, PendingRequests


def setup_seed(seed):
    torch.manual_seed(seed)


def parse_range_string(s):
    try:
        ranges = s.split(",")
        result = []
        for r in ranges:
            start, end = map(int, r.split("-"))
            result.append((start, end))
        return result
    except:
        raise argparse.ArgumentTypeError("参数必须是形如 '1-2,3-4' 的范围字符串")


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def tensor_to_list(tensor: Optional[torch.Tensor]) -> List:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.float().cpu().detach().numpy().tolist()


def list_to_tensor(lst: Optional[List]) -> torch.Tensor:
    if lst is None:
        return None
    return torch.tensor(lst)


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
    model_path: str, is_local: str, logger, master_handler_port: int
) -> Tuple[AsyncEngine, TokenizerUtils, MasterHandler]:
    model, tok, num_layers = load_master_model(model_path, logger)
    if is_local:
        generator = LLMGenerator(LocalRPCManager(logger, model_path, num_layers), logger, model, tok)
        master_handler = None
    else:
        pending_requests = PendingRequests()
        master_handler = MasterHandler(logger, pending_requests)
        await master_handler.start(master_handler_port)

        generator = LLMGenerator(RPCManager(pending_requests), logger, model, tok)
    engine = AsyncEngine(logger, generator)
    return engine, tok, master_handler
