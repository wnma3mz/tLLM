import asyncio

from fastapi import FastAPI, HTTPException
from shared_memory import RingBuffer

# api_server.py
app = FastAPI()
ring_buffer = RingBuffer("engine_buffer")
response_buffer = RingBuffer("response_buffer")


@app.post("/process")
async def process_request(data: dict):
    # 写入请求
    if not ring_buffer.write(str(data).encode()):
        raise HTTPException(status_code=503, detail="Buffer full")

    # 等待响应
    for _ in range(1000):  # 设置超时
        response = response_buffer.read()
        if response:
            return {"result": response.decode()}
        await asyncio.sleep(0.0001)

    raise HTTPException(status_code=408, detail="Request timeout")


# cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    ring_buffer.close()
    response_buffer.close()
