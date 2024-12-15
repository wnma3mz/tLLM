import argparse
import asyncio
from dataclasses import dataclass
import logging

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="MLX", choices=["MLX", "TORCH", "mlx", "torch"])
    return parser.parse_args()


args = parse_args()
import os

os.environ["TLLM_BACKEND"] = args.backend.upper()

from tllm.commons.manager import load_master_model
from tllm.engine import AsyncEngine
from tllm.entrypoints.image_server.image_protocol import Text2ImageRequest
from tllm.entrypoints.image_server.server_image import ImageServing
from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.generate import ImageGenerator, LLMGenerator
from tllm.img_helper import base64_to_pil_image
from tllm.network.manager import LocalRPCManager
from tllm.utils import setup_logger


@dataclass
class Args:
    model_path: str = "/Users/lujianghu/Documents/Llama-3.2-3B-Instruct"
    # model_path: str = "/Users/lujianghu/Documents/flux/schnell_4bit"
    # model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    is_debug: bool = False


def init_engine(model_path, logger):
    model, tok = load_master_model(model_path)
    rpc_manager = LocalRPCManager(model_path)
    generator = LLMGenerator(rpc_manager, logger, model, tok)
    engine = AsyncEngine(logger, generator)
    return engine, tok


def init_image_engine(model_path, logger):
    model, tok = load_master_model(model_path)
    rpc_manager = LocalRPCManager(model_path)
    generator = ImageGenerator(rpc_manager, logger, model, tok)
    engine = AsyncEngine(logger, generator)
    return engine


async def llm_generate():
    args = Args()

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok = init_engine(args.model_path, logger)
    _ = await engine.start()
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    openai_serving_chat = OpenAIServing(engine, tok, args)

    request = ChatCompletionRequest(model="test", messages=messages)
    response = await openai_serving_chat.create_chat_completion(request, None)
    print(response)


async def mllm_generate():
    args = Args()

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok = init_engine(args.model_path, logger)
    _ = await engine.start()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image_url", "image_url": {"file_path": "asserts/image-1.png"}},
            ],
        }
    ]
    openai_serving_chat = OpenAIServing(engine, tok, args)

    request = ChatCompletionRequest(model="test", messages=messages)
    response = await openai_serving_chat.create_chat_completion(request, None)
    print(response)


async def image_generate():
    args = Args()

    prompt = "a little dog"
    prompt = "germanic romanticism painting of an obscure winter forest in a geocore landscape. Ambient landscape lighting, heavy shading, crystal night sky, stunning stars, topography"
    config = {
        "num_inference_steps": 3,
        "height": 768,
        "width": 768,
    }

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine = init_image_engine(args.model_path, logger)
    _ = await engine.start()

    image_serving = ImageServing(engine, args)

    request = Text2ImageRequest(model="test", prompt=prompt, config=config)
    response = await image_serving.create_image(request, None)
    img = Image.open(base64_to_pil_image(response.base64))
    img.save("tt.png")


if __name__ == "__main__":
    asyncio.run(llm_generate())
    # asyncio.run(mllm_generate())
    # asyncio.run(image_generate())
