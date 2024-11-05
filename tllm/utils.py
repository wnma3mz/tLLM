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
from tllm.generate.llm_generator import LLMGenerator
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.register import HAS_MLX, MODEL_REGISTER
from tllm.rpc.manager import LocalRPCManager, RPCManager
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


def start_client(config_path: str, model_path: str) -> None:
    # 启动 client
    with open(config_path, "r") as f:
        config_list = json.load(f)

    os.system("rm -rf grpc_*.log")
    for pp_config in config_list:
        port = pp_config["url"].rsplit(":", 1)[-1]
        start_layer_idx, end_layer_idx = pp_config["layer_idx"]
        # TODO 启动远程服务
        if pp_config["tp_size"] > 1:
            cmd = f"torchrun --nproc_per_node={pp_config['tp_size']} --master_port={pp_config['master_port']} tllm/rpc/client.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"
        else:
            # 几乎等效于 torchrun --nproc_per_node=1
            cmd = f"python3 tllm/rpc/client.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"  #
        # 异步启动
        logger.info(f"begin start client {pp_config['pp_rank']}")
        os.popen(cmd)
        # 监听是否启动成功
        while True:
            if os.path.exists(f"grpc_{port}.log"):
                with open(f"grpc_{port}.log", "r") as f:
                    if "Starting gRPC server on port" in f.read():
                        break
            time.sleep(1)
        logger.info(f"start client {pp_config['pp_rank']} success")


def parse_url_list(config_path: str) -> List[str]:
    with open(config_path, "r") as f:
        config_list = json.load(f)
    return [pp_config["url"] for pp_config in config_list]


def init_engine(args) -> Tuple[AsyncEngine, TokenizerUtils]:
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    arch = config.architectures[0]
    if arch not in MODEL_REGISTER:
        raise ValueError(f"Model {arch} not supported")
    if HAS_MLX:
        arch = "MLX" + arch
    _, MY_CausalLM_CLASS, _ = MODEL_REGISTER[arch]

    url_list = parse_url_list(args.config_path)
    tok = TokenizerUtils(args.model_path)
    model = MY_CausalLM_CLASS.from_pretrained(logger, config, tok, args.model_path)
    if args.is_local:
        generator = LLMGenerator(LocalRPCManager(logger, args, config), logger, model)
    else:
        generator = LLMGenerator(RPCManager(url_list), logger, model)
    engine = AsyncEngine(logger, generator)
    return engine, tok


logger = setup_logger(__name__, logging.DEBUG)
# logger = setup_logger(__name__, logging.INFO)
