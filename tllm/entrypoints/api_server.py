import asyncio
import os
import time
from typing import Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from tllm.commons.manager import load_master_model
from tllm.engine import AsyncEngine
from tllm.entrypoints.image_server.image_protocol import Text2ImageRequest, Text2ImageResponse
from tllm.entrypoints.image_server.server_image import ImageServing
from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.entrypoints.utils import parse_master_args, serve_http, update_master_args
from tllm.generate import ImageGenerator, LLMGenerator
from tllm.network.manager import LocalRPCManager, RPCManager, WebsocketManager
from tllm.schemas import InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse
from tllm.singleton_logger import SingletonLogger
from tllm.utils import init_rpc_manager, setup_seed

openai_serving_chat: OpenAIServing = None
image_serving: ImageServing = None
ws_manager: WebsocketManager = None
rpc_manager: Union[RPCManager, LocalRPCManager]
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
    if not ws_manager.has_full_model and not args.is_local:
        raise ValueError("No available Full Node to process the request")
    if openai_serving_chat is None:
        raise ValueError("OpenAIServing instance is not initialized")
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
    if not ws_manager.has_full_model and not args.is_local:
        raise ValueError("No available Full Node to process the request")
    if openai_serving_chat is None:
        raise ValueError("OpenAIServing instance is not initialized")
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/create_image")
async def create_image(request: Text2ImageRequest, raw_request: Request) -> Text2ImageResponse:
    if not ws_manager.has_full_model and not args.is_local:
        raise ValueError("No available Full Node to process the request")
    if image_serving is None:
        raise ValueError("ImageServing instance is not initialized")
    try:
        generator = await image_serving.create_image(request, raw_request)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, Text2ImageResponse)
            return JSONResponse(content=generator.model_dump())
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=499)


@app.get("/health")
async def health(background_tasks: BackgroundTasks):
    # 检查是否需要重新更新节点的状态
    # 如果没有请求 health，那么状态不会被更新
    health_status = await rpc_manager.get_status()
    if len(health_status["last_check_result"]) > 0:
        logger.info(f"health check result: {health_status}")
        rpc_manager.stop_health_check()
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
        rpc_manager.update_url(host_list)
        await rpc_manager.send_config(f"{args.hostname}:{args.grpc_port}", host_list)
        # 后台持续进行健康检查，如果有节点挂掉，需要重新分配
        await rpc_manager.start_health_check()


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


async def init_app(engine: AsyncEngine, args):
    global app
    global logger, openai_serving_chat, image_serving
    logger = SingletonLogger.setup_master_logger()

    logger.info("args: %s", args)
    if args.is_image:
        image_serving = ImageServing(engine, args)
    else:
        openai_serving_chat = OpenAIServing(engine, args)
    return app


async def init_engine(args):
    setup_seed()
    logger = SingletonLogger.setup_master_logger()

    s1 = time.time()
    model = load_master_model(args.model_path)
    total_layers = model.num_layers  # 必须要有层数

    global ws_manager, rpc_manager

    ws_manager = WebsocketManager(total_layers, args.model_path, client_size=args.client_size)
    rpc_manager, master_handler = await init_rpc_manager(
        args.model_path, ws_manager.client_size, args.grpc_port, args.is_local
    )
    logger.info(f"Engine init Cost Time: {time.time() - s1:.4f}s. Total Layers: {total_layers}")
    if args.is_image:
        generator = ImageGenerator(rpc_manager, model)
    else:
        generator = LLMGenerator(rpc_manager, model)
    engine = AsyncEngine(generator)

    await engine.start()
    return engine


async def run_server(args) -> None:
    SingletonLogger.set_level("DEBUG" if args.is_debug else "INFO")
    args = update_master_args(args)

    engine = await init_engine(args)
    app = await init_app(engine, args)

    uvicorn_kwargs = {"host": ["::", "0.0.0.0"], "port": args.http_port, "timeout_graceful_shutdown": 5}
    shutdown_task = await serve_http(app, **uvicorn_kwargs)
    await shutdown_task


def main():
    global args
    args = parse_master_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
