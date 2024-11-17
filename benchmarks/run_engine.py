import asyncio
from dataclasses import dataclass
import logging

from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.utils import init_engine, setup_logger


@dataclass
class Args:
    # model_path: str = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    # model_path: str= "/Users/jianghulu/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
    # model_path: str = "/Users/jianghulu/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4"
    model_path: str = "/Users/jianghulu/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4"
    is_local: bool = True
    is_debug: bool = True


async def main():
    args = Args()

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok, _ = await init_engine(args.model_path, args.is_local, logger, master_handler_port=25111)
    _ = await engine.start()
    # messages = [{"role": "user", "content": "Hello, how are you?"}]
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
    generator = await openai_serving_chat.create_chat_completion(request, None)
    print(generator)


if __name__ == "__main__":
    asyncio.run(main())
