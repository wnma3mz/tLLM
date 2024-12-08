import asyncio
from dataclasses import dataclass
import logging

from PIL import Image

from tllm.entrypoints.image_protocol import Text2ImageRequest
from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.entrypoints.server_chat import OpenAIServing
from tllm.entrypoints.server_image import ImageServing
from tllm.generate import ImageGenerator, LLMGenerator
from tllm.img_helper import base64_to_pil_image
from tllm.utils import init_engine, setup_logger


@dataclass
class Args:
    model_path: str = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    # model_path: str = "/Users/lujianghu/Documents/flux/schnell_4bit"
    # model_path: str= "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
    # model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    is_local: bool = True
    is_debug: bool = True
    is_fake: bool = False


async def llm_generate():
    args = Args()

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok, _ = await init_engine(logger, args.model_path, 25111, args.is_local, args.is_fake, LLMGenerator)
    _ = await engine.start()
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    openai_serving_chat = OpenAIServing(engine, tok, args)

    request = ChatCompletionRequest(model="test", messages=messages)
    response = await openai_serving_chat.create_chat_completion(request, None)
    print(response)


async def mllm_generate():
    args = Args()

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, tok, _ = await init_engine(logger, args.model_path, 25111, args.is_local, args.is_fake, LLMGenerator)
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
    config = {
        "num_inference_steps": 2,
        "height": 512,
        "width": 512,
    }

    logger = setup_logger("engine", logging.DEBUG if args.is_debug else logging.INFO)
    engine, _, _ = await init_engine(logger, args.model_path, 25111, args.is_local, args.is_fake, ImageGenerator)
    _ = await engine.start()

    image_serving = ImageServing(engine, args)

    request = Text2ImageRequest(model="test", prompt=prompt, config=config)
    response = await image_serving.create_image(request, None)
    img = Image.open(base64_to_pil_image(response.base64))
    img.save("tt.png")


if __name__ == "__main__":
    asyncio.run(llm_generate())
    # asyncio.run(image_generate())
