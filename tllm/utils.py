import argparse
import json
import logging
import os
import socket
import time
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


def start_handler(config_path: str, model_path: str, logger) -> None:
    # 启动 handler
    with open(config_path, "r") as f:
        config = json.load(f)

    os.system("rm -rf grpc_*.log")
    for pp_config in config["client"]:
        port = pp_config["url"].rsplit(":", 1)[-1]
        start_layer_idx, end_layer_idx = pp_config["layer_idx"]
        # TODO 启动远程服务
        if pp_config["tp_size"] > 1:
            cmd = f"torchrun --nproc_per_node={pp_config['tp_size']} --master_port={pp_config['master_port']} tllm/rpc/handler.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"
        else:
            cmd = f"python3 tllm/rpc/handler.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"  #
        # 异步启动
        logger.info(f"begin start handler {pp_config['pp_rank']}")
        os.popen(cmd)
        # 监听是否启动成功
        while True:
            if os.path.exists(f"grpc_{port}.log"):
                with open(f"grpc_{port}.log", "r") as f:
                    if "Starting gRPC server on port" in f.read():
                        break
            time.sleep(1)
        logger.info(f"start handler {pp_config['pp_rank']} success")


def parse_config(config_path: str) -> Tuple[List[str], int]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return [x["url"] for x in config["client"]], int(config["master"]["url"].rsplit(":", 1)[-1])


async def init_engine(
    model_path: str, is_local: str, logger, url_list: Optional[List[str]] = None, master_handler_port: int = -1
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
        generator = LLMGenerator(LocalRPCManager(logger, model_path, config.num_hidden_layers), logger, model)
        master_handler = None
    else:
        if master_handler_port == -1:
            pending_requests = None
        else:
            pending_requests = PendingRequests()
            master_handler = MasterHandler(logger, pending_requests)
            await master_handler.start(master_handler_port)

        generator = LLMGenerator(RPCManager(url_list[0], pending_requests, len(url_list)), logger, model)
    engine = AsyncEngine(logger, generator)
    return engine, tok, master_handler
