import argparse
import asyncio
import json
import signal
from typing import Dict

from fastapi import FastAPI
import uvicorn

from tllm.singleton_logger import SingletonLogger


def parse_master_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--grpc_port", type=int, default=None)
    parser.add_argument("--http_port", type=int, default=8022)
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument(
        "--client_size",
        type=int,
        default=None,
        help="the number of the client, if not provided, will be parsed from the model path and auto calculated",
    )
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--is_image", action="store_true")
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


def load_master_config(config_path: str, args):
    with open(config_path, "r") as f:
        config = json.load(f)
    args.hostname = config["server"]["hostname"]
    args.http_port = config["server"]["http_port"]
    args.grpc_port = config["server"]["grpc_port"]
    args.client_size = len(config["client"])
    return args


def load_handler_config(config_path: str, args, idx: int):
    with open(config_path, "r") as f:
        config = json.load(f)
    args.grpc_port = config["client"][idx]["grpc_port"]
    args.hostname = config["client"][idx]["hostname"]
    args.master_addr = f'http://{config["server"]["hostname"]}:{config["server"]["http_port"]}'
    return args


async def serve_http(app: FastAPI, **uvicorn_kwargs: Dict):
    logger = SingletonLogger.setup_master_logger()

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    # asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    server_task = loop.create_task(server.serve())

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
