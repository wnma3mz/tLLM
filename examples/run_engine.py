import argparse
import asyncio
from dataclasses import dataclass
import math
import os
import time
from typing import List, Tuple

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="MLX", choices=["MLX", "TORCH", "mlx", "torch"])
    return parser.parse_args()


args = parse_args()
os.environ["TLLM_BACKEND"] = args.backend.upper()


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


@dataclass
class Args:
    # model_path: str = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    # model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    model_path: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    # model_path: str = "/Users/lujianghu/Documents/flux/schnell_4bit"
    # model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    is_debug: bool = False


class LocalRPCManager:
    # 并不发生通信，直接调用模型
    def __init__(self, model_path: str):
        self.model = load_client_model(0, math.inf, Communicator(), model_path)

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


def llm_message():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    return messages


def mllm_message():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image_url", "image_url": {"file_path": "asserts/image-1.png"}},
            ],
        }
    ]
    return messages


async def llm_generate(args, messages):
    engine = init_engine(args.model_path)
    await engine.start()
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    openai_serving_chat = OpenAIServing(engine, args)

    for _ in range(3):
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


if __name__ == "__main__":
    args = Args()
    asyncio.run(llm_generate(args, llm_message()))
    # asyncio.run(llm_generate(args, mllm_message()))
    # asyncio.run(image_generate(args))
