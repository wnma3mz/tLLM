import asyncio

from shared_memory import RingBuffer


async def run_engine():
    ring_buffer = RingBuffer("engine_buffer")
    response_buffer = RingBuffer("response_buffer")

    while True:
        # 非阻塞读取
        data = ring_buffer.read()
        if data:
            # 处理数据
            result = await process_data(data)
            # 写入响应
            response_buffer.write(result)
        await asyncio.sleep(0.0001)  # 极小的睡眠以避免CPU占用过高


async def process_data(data: bytes):
    # 模拟处理
    await asyncio.sleep(0.001)
    return b"Processed: " + data


if __name__ == "__main__":
    asyncio.run(run_engine())
