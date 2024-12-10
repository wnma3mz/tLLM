import argparse
import asyncio
import signal
from typing import Dict

from fastapi import FastAPI
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip_addr", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--grpc_port", type=int, default=None)
    parser.add_argument("--http_port", type=int, default=8022)
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--is_fake", action="store_true")
    return parser.parse_args()


async def serve_http(
    app: FastAPI, loop: asyncio.AbstractEventLoop, engine, master_handler, logger, **uvicorn_kwargs: Dict
):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    asyncio.set_event_loop(loop)
    server_task = loop.create_task(server.serve())

    # Setup graceful shutdown handlers
    async def shutdown_handler():
        server.should_exit = True

        try:
            await server.shutdown()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        if master_handler:
            try:
                await master_handler.stop()
            except Exception as e:
                logger.error(f"Error stopping master handler: {e}")

        try:
            await engine.stop()
        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
        finally:
            loop.stop()

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
