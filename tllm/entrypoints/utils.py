import argparse
import asyncio
import copy
import json
import signal
from typing import Dict

from fastapi import FastAPI
import uvicorn

from tllm.entrypoints.handler.handler import run
from tllm.network.helper import get_free_port, get_ips
from tllm.singleton_logger import SingletonLogger


def parse_master_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Specify the path of the model file or huggingface repo. Like mlx-community/Llama-3.2-1B-Instruct-bf16",
    )
    parser.add_argument("--hostname", type=str, help="The address of the client connection.")
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=None,
        help="Specify the port number used by the gRPC service. If this parameter is not provided, the default value (currently None, and the specific value may be determined by the program logic later) will be used.",
    )
    parser.add_argument(
        "--http_port",
        type=int,
        default=8022,
        help="Specify the port number used by the HTTP service. The default value is 8022, and this port can be modified by passing in a parameter according to actual needs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The path of the configuration file. If there is an additional configuration file to control the program's behavior, this parameter can be used to specify its path. By default, it is not specified.",
    )
    parser.add_argument(
        "--client_size",
        type=int,
        default=None,
        help="The number of clients. If this parameter is not provided, the program will try to parse and automatically calculate the number from the model path.",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="A boolean flag. If this parameter is specified in the command line, indicates that the model runs locally only",
    )
    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="A boolean flag used to turn on or off the debug mode. If this parameter is specified, the program will print more logs",
    )
    parser.add_argument(
        "--is_image",
        action="store_true",
        help="A boolean flag. The specific meaning start the Vincennes Diagram service",
    )
    return parser.parse_args()


def parse_handler_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpc_port", type=int, default=None, help="gRPC 服务的端口")
    parser.add_argument(
        "--master_addr", type=str, required=True, help="master 的 http 地址, 如 http://192.168.x.y:8022"
    )
    parser.add_argument("--hostname", type=str, default=None, help="提供给 master 连接的 ip, 如 192.168.x.y")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--client_idx", type=int, default=None, help="the client index in the config file")
    return parser.parse_args()


def update_master_args(args):
    if args.grpc_port is None:
        args.grpc_port = get_free_port()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        args.hostname = config["server"]["hostname"]
        args.http_port = config["server"]["http_port"]
        args.grpc_port = config["server"]["grpc_port"]
        args.client_size = len(config["client"])
    return args


def update_handler_args(args):
    if args.grpc_port is None:
        args.grpc_port = get_free_port()

    if args.config is not None:
        if args.client_idx is None:
            raise ValueError("client_idx is required when config is provided")
        with open(args.config_path, "r") as f:
            config = json.load(f)
        args.grpc_port = config["client"][args.client_idx]["grpc_port"]
        args.hostname = config["client"][args.client_idx]["hostname"]
        args.master_addr = f'http://{config["server"]["hostname"]}:{config["server"]["http_port"]}'

    # 如果指定了 hostname, 则只使用指定的 hostname
    if args.hostname is not None and isinstance(args.hostname, str):
        ip_addr_list = [args.hostname]
    else:
        ip_addr_list = get_ips()

    if len(ip_addr_list) == 0:
        raise ValueError("No available ip address")

    return args, ip_addr_list


async def serve_http(app: FastAPI, args, **uvicorn_kwargs: Dict):
    logger = SingletonLogger.setup_master_logger()

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    # asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    server_task = loop.create_task(server.serve())

    # 必须在 server 启动之后再启动 client，或者是异步的
    if args.is_local:
        args_handler = copy.deepcopy(args)
        args_handler.hostname = "localhost"
        args_handler.grpc_port = None
        args_handler.master_addr = f"http://{args_handler.hostname}:{args_handler.http_port}"
        await run(args_handler)

    # Setup graceful shutdown handlers
    async def shutdown_handler():
        server.should_exit = True

        try:
            await server.shutdown()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        # if master_handler:
        #     try:
        #         await master_handler.stop()
        #     except Exception as e:
        #         logger.error(f"Error stopping master handler: {e}")

        # try:
        #     await engine.stop()
        # except Exception as e:
        #     logger.error(f"Error stopping engine: {e}")
        # finally:
        #     loop.stop()

        logger.info("Shutdown sequence completed")

    async def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        await shutdown_handler()

    async def dummy_shutdown() -> None:
        pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(signal_handler()))

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down FastAPI HTTP server.")
        await shutdown_handler()
    except Exception as e:
        logger.error(f"Unexpected error in server task: {e}")
        await shutdown_handler()
    finally:
        return dummy_shutdown()
