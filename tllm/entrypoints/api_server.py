import argparse
import asyncio
import logging
import os
import signal
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.schemas import InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse
from tllm.utils import init_engine, setup_logger, setup_seed
from tllm.websocket.manager import PipelineManager, WebsocketManager

engine: None
openai_serving_chat: OpenAIServing = None
ws_manager: WebsocketManager = None


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "..", "static", "index.html")
    with open(html_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse:
    if not ws_manager.has_full_model and not is_local:
        raise ValueError("No available Full Node to process the request")
    try:
        generator = await openai_serving_chat.create_chat_completion(request, raw_request)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=499)


@app.post("/v1/completions")
async def create_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse:
    if not ws_manager.has_full_model and not is_local:
        raise ValueError("No available Full Node to process the request")
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.get("/health")
async def health(background_tasks: BackgroundTasks):
    # 检查是否需要重新更新节点的状态
    # 如果没有请求 health，那么状态不会被更新
    health_status = await pp_manager.get_status()
    if len(health_status["last_check_result"]) > 0:
        logger.info(f"health check result: {health_status}")
        pp_manager.stop_health_check()
        ws_manager.unset_connect_clients(health_status["last_check_result"])
        background_tasks.add_task(update_model_url)

    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():  #
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": 0.1}
    return JSONResponse(content=ver)


@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    # 前端页面展示用
    await websocket.accept()
    ws_manager.monitor_websockets.add(websocket)
    try:
        await websocket.send_json(ws_manager.get_state())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.monitor_websockets.remove(websocket)


async def update_model_url():
    # 如果有操作需要更新各个客户端的信息，需要在这里更新
    if ws_manager.has_full_model:
        return
    host_list = ws_manager.set_connect_clients()
    if len(host_list) > 0:
        clients = ws_manager.connect_clients
        openai_serving_chat.engine.update_url(clients[0].host, len(clients))
        pp_manager.update_url(host_list)
        await pp_manager.send_config(args.master_url, host_list)
        # 后台持续进行健康检查，如果有节点挂掉，需要重新分配
        await pp_manager.start_health_check()


@app.post("/register_client")
async def register_client_func(
    request: RegisterClientRequest, raw_request: Request, background_tasks: BackgroundTasks
) -> RegisterClientResponse:
    model_path: str = args.model_path
    response = await ws_manager.register_client(request, model_path)
    if request.pp_rank == -1:
        logger.info(f"Client {request.client_id} register")
    else:
        logger.info(f"Client {request.client_id} has been init: {request}")
        background_tasks.add_task(update_model_url)
    return response


@app.post("/init_model")
async def init_model_func(
    request: InitModelRequest, raw_request: Request, background_tasks: BackgroundTasks
) -> InitModelResponse:
    logger.info(f"Client {request.client_id} init: {request}")
    response = await ws_manager.init_client(request)
    background_tasks.add_task(update_model_url)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_url", type=str, required=True)
    parser.add_argument("--master_handler_port", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--is_fake", action="store_true")
    return parser.parse_args()


async def serve_http(app: FastAPI, loop: asyncio.AbstractEventLoop, master_handler, **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    asyncio.set_event_loop(loop)
    server_task = loop.create_task(server.serve())

    # Setup graceful shutdown handlers
    async def shutdown_handler():
        server.should_exit = True

        if master_handler:
            try:
                await master_handler.stop()
            except Exception as e:
                logger.error(f"Error stopping master handler: {e}")

        try:
            await engine.stop()
            await server.shutdown()
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
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Shutting down FastAPI HTTP server.")
        await shutdown_handler()
    except Exception as e:
        logger.error(f"Unexpected error in server task: {e}")
        await shutdown_handler()
        raise


async def run_server(args) -> None:
    setup_seed(42)
    global app
    global logger, engine, ws_manager, pp_manager, openai_serving_chat
    global is_local
    is_local = args.is_local

    logger = setup_logger("master", logging.DEBUG if args.is_debug else logging.INFO)

    logger.info("args: %s", args)

    s1 = time.time()
    engine, tok, master_handler = await init_engine(
        logger, args.model_path, args.master_handler_port, args.is_local, args.is_fake
    )

    logger.info(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(engine, tok, args)
    total_layers = engine.generator.model.num_layers
    ws_manager = WebsocketManager(total_layers, args.model_path)
    pp_manager = PipelineManager(ws_manager.client_size)

    loop = await engine.start()
    uvicorn_kwargs = {"host": ["::", "0.0.0.0"], "port": args.port}
    shutdown_task = await serve_http(app, loop, master_handler, **uvicorn_kwargs)
    await shutdown_task


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
