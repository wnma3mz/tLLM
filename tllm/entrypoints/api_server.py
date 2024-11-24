import argparse
import asyncio
import logging
import os
import signal
import time
from typing import *

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.schemas import ClientData, InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse
from tllm.utils import init_engine, setup_logger, setup_seed
from tllm.websocket.manager import WebsocketManager

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


@app.post("/health")
async def health():
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
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


def update_master_url(clients: List[ClientData]):
    # 客户端断开连接后，需要重新更新 url
    if clients:
        logger.info(f"Update url: {clients[0].host}; PP Size: {len(clients)}")
        openai_serving_chat.engine.update_url(clients[0].host, len(clients))


async def update_model_url():
    # 如果有操作需要更新各个客户端的信息，需要在这里更新
    if ws_manager.has_full_model:
        return
    logger.info("Update model url start")
    ws_manager.set_connect_clients()
    update_master_url(ws_manager.connect_clients)
    await ws_manager.send_config(args.master_url)
    logger.info("Update model url end")


@app.post("/register_client")
async def register_client_func(request: RegisterClientRequest, raw_request: Request) -> RegisterClientResponse:
    model_path: str = args.model_path
    response = await ws_manager.register_client(request, model_path)
    if request.pp_rank == -1:
        logger.info(f"Client {request.client_id} register")
    else:
        logger.info(f"Client {request.client_id} has been init: {request}")
        await update_model_url()
    return response


@app.post("/init_model")
async def init_model_func(request: InitModelRequest, raw_request: Request) -> InitModelResponse:
    logger.info(f"Client {request.client_id} init: {request}")
    response = await ws_manager.init_client(request)
    await update_model_url()
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_url", type=str, required=True)
    parser.add_argument("--master_handler_port", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    return parser.parse_args()


async def serve_http(app: FastAPI, loop: asyncio.AbstractEventLoop, master_handler, **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    asyncio.set_event_loop(loop)
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        logger.info("Shutting down FastAPI HTTP server.")
        if master_handler:
            await master_handler.stop()
        await engine.stop()
        return server.shutdown()


async def run_server(args) -> None:
    setup_seed(42)
    global app
    global logger, engine, ws_manager, openai_serving_chat
    global is_local
    is_local = args.is_local

    logger = setup_logger("master", logging.DEBUG if args.is_debug else logging.INFO)

    logger.info("args: %s", args)

    s1 = time.time()
    engine, tok, master_handler = await init_engine(args.model_path, args.is_local, logger, args.master_handler_port)

    logger.info(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(engine, tok, args)
    total_layers = engine.generator.model.num_layers
    ws_manager = WebsocketManager(total_layers, args.model_path)

    loop = await engine.start()
    uvicorn_kwargs = {"host": "0.0.0.0", "port": args.port}
    shutdown_task = await serve_http(app, loop, master_handler, **uvicorn_kwargs)
    await shutdown_task


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
