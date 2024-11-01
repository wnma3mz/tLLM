import argparse
from contextlib import asynccontextmanager
import os
import time
from typing import *

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from tllm.entrypoints.layer_manager import LayerManager
from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.utils import init_engine, logger, setup_seed, start_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时开始处理队列
    await engine.start()
    yield
    # 关闭时停止处理队列
    await engine.stop()


app = FastAPI(lifespan=lifespan)
openai_serving_chat: OpenAIServing

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
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


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
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--need_start_client", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(42)
    args = parse_args()

    engine, tok = init_engine(args)

    s1 = time.time()
    if args.need_start_client:
        start_client(logger, args.config_path, args.model_path)
    logger.info(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(engine, tok, args)

    model_name = openai_serving_chat.model_name
    total_layers = engine.model.num_layers
    layer_manager = LayerManager(total_layers=total_layers, model_name=model_name)

    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
