import asyncio
from dataclasses import dataclass
import logging

from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.utils import init_engine, setup_logger


@dataclass
class Args:
    model_path: str = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    is_local: bool = True
    is_debug: bool = True


async def main():
    args = Args()

    logger = setup_logger(__name__, logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok, _ = await init_engine(args.model_path, args.is_local, logger)
    _ = await engine.start()
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    openai_serving_chat = OpenAIServing(engine, tok, args)

    request = ChatCompletionRequest(model="test", messages=messages)
    generator = await openai_serving_chat.create_chat_completion(request, None)
    print(generator)


if __name__ == "__main__":
    asyncio.run(main())
