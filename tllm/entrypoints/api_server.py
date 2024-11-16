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
from tllm.entrypoints.websocket_manager import WebsocketManager
from tllm.utils import init_engine, setup_logger, setup_seed

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
    if ws_manager.url_list is None:
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


@app.get("/unregister_layer_idx")
async def get_unregister_layer_idx() -> Dict[str, List[int]]:
    return JSONResponse(content={"data": ws_manager.unregister_layer_idx()})


@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    await websocket.accept()
    ws_manager.monitor_websockets.add(websocket)
    try:
        await websocket.send_json(ws_manager.get_state())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.monitor_websockets.remove(websocket)


async def update_state():
    # 当 url_list 不为空时，表示可以处理服务
    # - master 端更新 url
    # - 发送给所有 pp，告知转发的 url 是什么（forward url）
    if ws_manager.url_list is not None:
        openai_serving_chat.engine.generator.manager.update_url(ws_manager.url_list[0], len(ws_manager.url_list))
        for i, client_id in enumerate(ws_manager.client_id_list):
            if i == len(ws_manager.client_id_list) - 1:
                url = ws_manager.master_url
            else:
                url = ws_manager.url_list[i + 1]
            message = {"type": "forward_url", "forward_url": url, "pp_rank": i, "master_url": ws_manager.master_url}
            await ws_manager.ws_clients[client_id].send_json(message)
    else:
        return None


@app.websocket("/ws/client/{client_id}")
async def client_websocket(websocket: WebSocket, client_id: str):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "register_layers":
                ws_manager.register_client(client_id, data, websocket)

                await ws_manager.broadcast_state()
                # 根据 layer idx 自动算 pp idx
                ws_manager.update_pp_url_list()
                await update_state()

    except WebSocketDisconnect:
        # 如果断开连接，删除client
        ws_manager.unregister_client(client_id)
        ws_manager.update_pp_url_list()
        await update_state()
        await ws_manager.broadcast_state()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_url", type=str, required=True)
    parser.add_argument("--master_handler_port", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, default=None)
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

    logger = setup_logger(__name__, logging.DEBUG if args.is_debug else logging.INFO)

    logger.info("args: %s", args)

    s1 = time.time()
    engine, tok, master_handler = await init_engine(args.model_path, args.is_local, logger, args.master_handler_port)

    logger.info(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(engine, tok, args)
    model_name = openai_serving_chat.model_name
    total_layers = engine.generator.model.num_layers
    ws_manager = WebsocketManager(total_layers, model_name, args.master_url)

    loop = await engine.start()
    uvicorn_kwargs = {"host": "0.0.0.0", "port": args.port}
    shutdown_task = await serve_http(app, loop, master_handler, **uvicorn_kwargs)
    await shutdown_task


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
