import argparse
import asyncio
import base64
import json
import math
import time
from typing import List, Tuple

from PIL import Image
import numpy as np

from tllm.commons.tp_communicator import Communicator
from tllm.commons.weight_manager import load_client_model, load_master_model
from tllm.engine import AsyncEngine
from tllm.entrypoints.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.generate import LLMGenerator
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


def init_engine(model_path: str) -> AsyncEngine:
    model = load_master_model(model_path)
    rpc_manager = LocalRPCManager(model_path)
    generator = LLMGenerator(rpc_manager, model)
    engine = AsyncEngine(generator)
    return engine


async def llm_generate(args, messages, is_gen_image: bool = False):
    engine = init_engine(args.model_path)
    await engine.start()
    openai_serving_chat = OpenAIServing(engine, args)

    if is_gen_image:
        max_tokens = 576
        add_generation_prompt = False
    else:
        max_tokens = 100
        add_generation_prompt = True

    request = ChatCompletionRequest(
        model="test",
        messages=messages,
        max_tokens=max_tokens,
        add_generation_prompt=add_generation_prompt,
        is_gen_image=is_gen_image,
    )
    response: ChatCompletionResponse = await openai_serving_chat.create_chat_completion(request, None)

    if is_gen_image:
        img_bytes = base64.b64decode(response.choices[0].message.content)
        Image.fromarray(np.frombuffer(img_bytes, dtype=np.uint8).reshape(384, 384, 3)).save("tt.png")
    else:
        print(response)


def load_message():
    with open("asserts/messages.json", "r") as f:
        messages_dict = json.load(f)
    return messages_dict


def gen_img_message():
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "assistant", "content": "<begin_of_image>"},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--message_type", type=str, default="llm", choices=["llm", "mllm", "image"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    messages_dict = load_message()
    if args.message_type == "llm":
        asyncio.run(llm_generate(args, messages_dict["llm"][0]))
    elif args.message_type == "mllm":
        asyncio.run(llm_generate(args, messages_dict["mllm"][0]))
    elif args.message_type == "image":
        message = gen_img_message()
        asyncio.run(llm_generate(args, message, True))
    else:
        raise ValueError(f"Unknown message type: {args.message_type}")
