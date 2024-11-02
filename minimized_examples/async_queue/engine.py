import asyncio
import logging
import random
from typing import *

from tllm.models.protocol import SequenceRequestData

random.seed(42)

logging.basicConfig(level=logging.INFO)


async def generate_func(sequence_requests_list: List[SequenceRequestData]):
    # 无需返回，直接在这里设置相关数据
    for sequence_request in sequence_requests_list:
        if sequence_request.output_ids is None:
            sequence_request.output_ids = [1]
            sequence_request.output_text = "1"
        else:
            sequence_request.output_ids.append(1)
            sequence_request.output_text += "1"
        if len(sequence_request.output_ids) > 10 or random.random() > 0.5:
            sequence_request.is_stop = True


class AsyncEngine:
    def __init__(self, model):
        self.model = model
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        self.limit_size: int = 5  # 每次最多处理 5 个请求，prefill + decode

    async def fetch_data(self):
        # prefill 队列和 decoding 队列的调度逻辑
        sequence_data_list = []

        # 优先从 decoding_queue 取数据
        while not self.decoding_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.decoding_queue.get()
            sequence_data_list.append(sequence_data)

        # 从 prefill_queue 中取数据，直到达到限制
        while not self.prefill_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.prefill_queue.get()
            sequence_data_list.append(sequence_data)

        return sequence_data_list

    async def _generate(self):
        while True:
            sequence_data_list: List[SequenceRequestData] = await self.fetch_data()
            try:
                await generate_func(sequence_data_list)

                for sequence_data in sequence_data_list:
                    if not sequence_data.is_stop:
                        await self.decoding_queue.put(sequence_data)
                    async with sequence_data.condition:
                        sequence_data.condition.notify()

                # await asyncio.sleep(0.1)
            except Exception as e:
                logging.info(f"Error processing prefill_queue data: {str(e)}")
            except BaseException as e:
                logging.info(f"BaseException Error processing prefill_queue data: {str(e)}")
            finally:
                await asyncio.sleep(0.1)

    async def generate_stream(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
                    yield data  # 流式返回数据的内容，可以控制

        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")

    async def generate(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
                    # 这里可以进行你需要的处理，例如更新输出
                    # 确保在这里将 output_text 更新
            return data  # 返回最终的数据对象
        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")

    async def start(self):
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._generate())

    async def stop(self):
        logging.info("Stopping processing sequence_data")
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None


async def test_process(test_data, stream: bool = True):
    if stream:
        async for result in engine.generate_stream(test_data):
            print(f"Request {test_data.request_id} received: {result}")
    else:
        result = await engine.generate(test_data)
    return result


async def main():
    global engine
    engine = AsyncEngine(None)
    await engine.start()

    test_data_list = [
        SequenceRequestData(request_id="123"),
        SequenceRequestData(request_id="456"),
        SequenceRequestData(request_id="789"),
    ]

    stream = False
    results = await asyncio.gather(*(test_process(data, stream) for data in test_data_list))
    if not stream:
        for result in results:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
