import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import os
import signal
import time
from typing import *

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from tllm.entrypoints.layer_manager import LayerManager
from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.utils import init_engine, parse_url_list, setup_logger, setup_seed, start_handler

engine: None
openai_serving_chat: OpenAIServing = None
layer_manager: LayerManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(lifespan=lifespan)


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
    return JSONResponse(content={"data": layer_manager.unregister_layer_idx()})


@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    await websocket.accept()
    layer_manager.monitor_websockets.add(websocket)
    try:
        await websocket.send_json(layer_manager.get_state())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        layer_manager.monitor_websockets.remove(websocket)


@app.websocket("/ws/client/{client_id}")
async def client_websocket(websocket: WebSocket, client_id: str):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "register_layers":
                layer_manager.register_client(client_id, data)

                await layer_manager.broadcast_state()
                # 根据 layer idx 自动算 pp idx
                url_list = layer_manager.get_pp_url_list()
                if url_list is not None:
                    for pp_idx, url in enumerate(url_list):
                        openai_serving_chat.engine.model.server.update_url(pp_idx, url)
    except WebSocketDisconnect:
        # 如果断开连接，删除client
        pp_idx = layer_manager.unregister_client(client_id)
        # 删除
        if pp_idx != -1:
            openai_serving_chat.engine.model.server.remove_url(pp_idx)
        await layer_manager.broadcast_state()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--need_start_handler", action="store_true")
    parser.add_argument("--is_local", action="store_true")
    parser.add_argument("--is_debug", action="store_true")
    return parser.parse_args()


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
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
        return server.shutdown()


async def run_server(args) -> None:
    setup_seed(42)
    global app
    global logger, engine, layer_manager, openai_serving_chat

    logger = setup_logger(__name__, logging.DEBUG if args.is_debug else logging.INFO)

    if args.need_start_handler:
        start_handler(args.config_path, args.model_path, logger)

    logger.info("args: %s", args)

    s1 = time.time()
    url_list = None if args.config_path is None else parse_url_list(args.config_path)
    engine, tok = init_engine(args.model_path, args.is_local, logger, url_list)

    logger.info(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(engine, tok, args)
    model_name = openai_serving_chat.model_name
    total_layers = engine.generator.model.num_layers
    layer_manager = LayerManager(total_layers=total_layers, model_name=model_name)

    uvicorn_kwargs = {"host": "0.0.0.0", "port": args.port}
    shutdown_task = await serve_http(app, **uvicorn_kwargs)
    await shutdown_task


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
