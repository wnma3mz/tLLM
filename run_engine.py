import argparse
import asyncio
import json
import math
import os
import time
from typing import List, Tuple

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="MLX", choices=["MLX", "TORCH"])
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="AUTO",
        choices=["AUTO", "TORCH", "VLLM", "XFormers"],
        help="Attention backend if backend is TORCH",
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--message_type", type=str, default="llm", choices=["llm", "mllm", "image"])
    return parser.parse_args()


args = parse_args()
os.environ["TLLM_BACKEND"] = args.backend
os.environ["TLLM_ATTN_BACKEND"] = args.attn_backend

from tllm.commons.manager import load_client_model, load_master_model
from tllm.commons.tp_communicator import Communicator
from tllm.engine import AsyncEngine
from tllm.entrypoints.image_server.image_protocol import Text2ImageRequest
from tllm.entrypoints.image_server.server_image import ImageServing
from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.generate import ImageGenerator, LLMGenerator
from tllm.img_helper import base64_to_pil_image
from tllm.schemas import MIX_TENSOR, SeqInput
from tllm.singleton_logger import SingletonLogger

SingletonLogger.set_level("INFO")
logger = SingletonLogger.setup_master_logger()


class LocalRPCManager:
    # 并不发生通信，直接调用模型
    def __init__(self, model_path: str):
        self.model = load_client_model(0, math.inf, Communicator(logger), model_path)

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        return output_hidden_states, [time.perf_counter() - s1]

    async def image_forward(
        self,
        hidden_states: MIX_TENSOR,
        text_embeddings: MIX_TENSOR,
        seq_len: int,
        height: int,
        width: int,
        request_id: str,
    ) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, text_embeddings, seq_len, height, width, [request_id])
        return output_hidden_states, [time.perf_counter() - s1]


def init_engine(model_path: str) -> AsyncEngine:
    model = load_master_model(model_path)
    rpc_manager = LocalRPCManager(model_path)
    generator = LLMGenerator(rpc_manager, model)
    engine = AsyncEngine(generator)
    return engine


def init_image_engine(model_path: str) -> AsyncEngine:
    model = load_master_model(model_path)
    rpc_manager = LocalRPCManager(model_path)
    generator = ImageGenerator(rpc_manager, model)
    engine = AsyncEngine(generator)
    return engine


async def llm_generate(args, messages):
    engine = init_engine(args.model_path)
    await engine.start()
    openai_serving_chat = OpenAIServing(engine, args)

    request = ChatCompletionRequest(model="test", messages=messages, max_tokens=100)
    response = await openai_serving_chat.create_chat_completion(request, None)
    print(response)


async def image_generate(args):
    prompt = "germanic romanticism painting of an obscure winter forest in a geocore landscape. Ambient landscape lighting, heavy shading, crystal night sky, stunning stars, topography"
    config = {
        "num_inference_steps": 3,
        "height": 768,
        "width": 768,
    }

    engine = init_image_engine(args.model_path)
    await engine.start()

    image_serving = ImageServing(engine, args)

    request = Text2ImageRequest(model="test", prompt=prompt, config=config)
    response = await image_serving.create_image(request, None)
    img = Image.open(base64_to_pil_image(response.base64))
    img.save("tt.png")


def load_message():
    with open("asserts/messages.json", "r") as f:
        messages_dict = json.load(f)
    return messages_dict


if __name__ == "__main__":
    args = parse_args()
    messages_dict = load_message()
    if args.message_type == "llm":
        asyncio.run(llm_generate(args, messages_dict["llm"][0]))
    elif args.message_type == "mllm":
        asyncio.run(llm_generate(args, messages_dict["mllm"][0]))
    elif args.message_type == "image":
        asyncio.run(image_generate(args))
    else:
        raise ValueError(f"Unknown message type: {args.message_type}")
