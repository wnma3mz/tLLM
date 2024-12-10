import asyncio
import os
import time
from typing import AsyncIterator

from fastapi import HTTPException, Request

from tllm.engine import AsyncEngine, ImageGenerator
from tllm.entrypoints.image_server.image_protocol import Text2ImageRequest, Text2ImageResponse
from tllm.entrypoints.protocol import random_uuid
from tllm.schemas import ImageRequestData, RequestOutput


def create_error_response(message: str) -> Text2ImageResponse:
    raise HTTPException(status_code=499, detail=message)


class ImageServing:
    def __init__(self, engine: AsyncEngine, args):
        self.engine = engine
        self.model_name = os.path.basename(args.model_path)

    async def create_image(self, request: Text2ImageRequest, raw_request: Request):
        request_id = f"chat-{random_uuid()}"

        request_data = ImageRequestData(
            request_id=request_id,
            prompt=request.prompt,
            config=request.config,
            seed=request.seed,
        )
        result_generator = self.engine.generate_stream(request_data)

        if request.stream:
            raise NotImplementedError("Stream is not supported for image generation.")
            return self.image_full_stream_generator(request, raw_request, request_id, result_generator)
        else:
            return await self.image_full_generator(request, raw_request, request_id, result_generator)

    async def image_full_generator(
        self, request: Text2ImageRequest, raw_request: Request, request_id: str, result_generator: AsyncIterator
    ) -> Text2ImageResponse:
        final_res = None
        created_time = int(time.time())
        try:
            async for res in result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(request_id)
                    create_error_response("Client disconnected")
                await asyncio.sleep(0.01)
                final_res: RequestOutput = res
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            create_error_response("Client disconnected")

        output = final_res.outputs[0]
        return Text2ImageResponse(base64=output)
