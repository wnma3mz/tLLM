import asyncio
import json
from typing import *

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()


class Engine:
    async def generate(self) -> AsyncGenerator[int, None]:
        i = 0
        while True:
            if i >= 5:
                break
            i += 1
            await asyncio.sleep(0.1)  # 使用异步延迟
            yield i
        yield "end"


app.engine = Engine()


@app.post("/generate")
async def stream_response(request: Request):
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    results_generator = app.engine.generate()

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            ret = {"text": request_output}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())
    else:
        final_output = []
        final_res = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                return Response(status_code=499)
            final_output.append(request_output)
            final_res = request_output
        ret = {"text": final_output}
        return JSONResponse(ret)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
