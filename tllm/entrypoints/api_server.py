import argparse
import logging
import time
from typing import *

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing, start_client
from tllm.utils import setup_seed

app = FastAPI()

openai_serving_chat: OpenAIServing


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse:
    # logging.info("messages:", request.messages)
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


@app.post("/status")
async def status(request: Dict[str, float]):
    cost_time = request.get("cost_time", 0)
    pp_rank = request.get("pp_rank", 0)
    return {"cost_time": cost_time, "pp_rank": pp_rank}


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": 0.1}
    return JSONResponse(content=ver)


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

    s1 = time.time()
    if args.need_start_client:
        start_client(args.config_path, args.model_path)
    print(f"init cost time {time.time() - s1}")
    openai_serving_chat = OpenAIServing(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
