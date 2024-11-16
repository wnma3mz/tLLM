import argparse
import logging
import socket
from typing import *

import torch
from transformers import AutoConfig

from tllm.engine import AsyncEngine
from tllm.generate import LLMGenerator, TokenizerUtils
from tllm.models.register import HAS_MLX, MODEL_REGISTER
from tllm.rpc.manager import LocalRPCManager, RPCManager
from tllm.rpc.master_handler import MasterHandler, PendingRequests
from tllm.schemas import NodeConfig


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


# 用于 RPC 请求
def call_remote_init(model_rref, node_config: NodeConfig) -> torch.futures.Future:
    return model_rref.rpc_async().init_model(node_config)


def call_remote_forward(
    model_rref, hidden_states: Optional[torch.Tensor], shape_hidden_states: Tuple[int], uuid: str
) -> torch.futures.Future:
    return model_rref.rpc_async().forward(hidden_states, shape_hidden_states, uuid)


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
    if model_path.endswith(".gguf"):
        raise ValueError("GGUF model not supported")
        arch = "MLXLlamaForCausalLM"
        from tllm.models.gguf_utils import load_gguf_weight

        state_dict, config, _ = load_gguf_weight(model_path)
        tok_path = ...
        tok = TokenizerUtils(tok_path)
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = config.architectures[0]
        if HAS_MLX:
            arch = "MLX" + arch
        if arch not in MODEL_REGISTER:
            raise ValueError(f"Model {arch} not supported")
        tok = TokenizerUtils(model_path)
        state_dict = None

    MY_CausalLM_CLASS, _ = MODEL_REGISTER[arch]

    model = MY_CausalLM_CLASS.from_pretrained(logger, config, model_path, state_dict)
    if is_local:
        generator = LLMGenerator(LocalRPCManager(logger, model_path, config.num_hidden_layers), logger, model, tok)
        master_handler = None
    else:
        pending_requests = PendingRequests()
        master_handler = MasterHandler(logger, pending_requests)
        await master_handler.start(master_handler_port)

        generator = LLMGenerator(RPCManager(pending_requests), logger, model, tok)
    engine = AsyncEngine(logger, generator)
    return engine, tok, master_handler
