import asyncio
import time
import traceback
from typing import Callable, List, Optional, Union

from tllm.generate import ImageGenerator, LLMGenerator
from tllm.schemas import SequenceRequestData
from tllm.singleton_logger import SingletonLogger


class AsyncEngine:
    def __init__(self, generator: Union[LLMGenerator, ImageGenerator], sleep_time: float = 0.0, limit_size: int = 5):
        self.generator = generator
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.abort_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        self.limit_size: int = limit_size  # 每次最多处理 5 个请求，prefill + decode
        self.sleep_time: float = sleep_time
        self.logger = SingletonLogger.setup_master_logger()
        self.queue_not_empty: asyncio.Event = asyncio.Event()
        self._loop = None

    @property
    def tok(self):
        return self.generator.tok

    @property
    def process_mm_input(self) -> Optional[Callable]:
        return getattr(self.generator.model, "process_mm_input", None)

    async def fetch_data(self):
        aborting_request_ids = set()
        while not self.abort_queue.empty():
            request_id = self.abort_queue.get_nowait()
            aborting_request_ids.add(request_id)

        async def process_queue(queue: asyncio.Queue) -> List[SequenceRequestData]:
            processed_count = 0
            temp_list = []
            while not queue.empty() and len(request_data_list) + processed_count < self.limit_size:
                request_data = queue.get_nowait()
                if request_data.request_id in aborting_request_ids:
                    self.logger.debug(f"Aborting generate request_id: {request_data.request_id}")
                    request_data.is_stop = True
                    request_data.finish_reason_list = ["abort"]
                    async with request_data.condition:
                        request_data.condition.notify()
                    aborting_request_ids.remove(request_data.request_id)  # Avoid processing again
                else:
                    temp_list.append(request_data)
                    processed_count += 1
            return temp_list

        request_data_list = []
        # 优先从 decoding_queue 取数据
        request_data_list += await process_queue(self.decoding_queue)
        request_data_list += await process_queue(self.prefill_queue)
        return request_data_list

    async def _run_generation_batch(self, request_data_list: List[SequenceRequestData]):
        try:
            await self.generator.generate(request_data_list)

            for request_data in request_data_list:
                if not request_data.is_stop:
                    self.decoding_queue.put_nowait(request_data)
                    self.queue_not_empty.set()  # Ensure the loop wakes up if it was waiting

        except (asyncio.CancelledError, asyncio.TimeoutError) as e:
            self.logger.error(f"{type(e).__name__}")
            for request_data in request_data_list:
                request_data.is_normal_process = False
            traceback.print_exc()
        except Exception as e:
            self.logger.error(f"Error processing prefill_queue data: {repr(e)}")
            self.logger.error(f"Error request_id: {','.join(x.request_id for x in request_data_list)}")
            traceback.print_exc()
        finally:
            for request_data in request_data_list:
                async with request_data.condition:
                    request_data.condition.notify()

    async def _generate(self):
        while True:
            request_data_list: List[SequenceRequestData] = await self.fetch_data()
            if not request_data_list:
                await self.queue_not_empty.wait()
                continue

            if self._loop is not None:
                self._loop.create_task(self._run_generation_batch(request_data_list))
            else:
                self.logger.error("Event loop not available for creating generation task.")
                # Handle error appropriately, maybe notify requests with an error state

            # Clear the event if both queues might be empty now,
            # allowing the loop to wait if needed next iteration.
            # The event will be set again if new items are added or put back.
            if self.prefill_queue.empty() and self.decoding_queue.empty():
                self.queue_not_empty.clear()

    async def generate_stream(self, request_data: SequenceRequestData):
        self.prefill_queue.put_nowait(request_data)
        self.queue_not_empty.set()

        try:
            async with request_data.condition:
                while not request_data.is_stop:
                    await asyncio.wait_for(request_data.condition.wait(), request_data.timeout)
                    # 流式返回数据的内容
                    yield request_data.to_request_output()
                if getattr(request_data, "ttft_cost_time", None) is not None:
                    self.logger.info(
                        f"[request_id] {request_data.request_id}] ttft: {request_data.ttft_cost_time:.4f} s"
                    )
                    self.logger.info(
                        f"[request_id] {request_data.request_id}] tpot: {(len(request_data.output_ids) - 1) / (time.perf_counter() - request_data.decode_start_ts):.4f} token/s"
                    )
                # Need it?
                yield request_data.to_request_output()
        except Exception as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownException")

    async def start(self) -> asyncio.AbstractEventLoop:
        if self.processing_task is not None:
            raise RuntimeError("Engine is already running")

        self._loop = asyncio.get_running_loop()
        self.processing_task = self._loop.create_task(self._generate())
        return self._loop

    async def stop(self):
        self.logger.info("Stopping processing request_data")
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await asyncio.wait_for(self.processing_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.debug("Task cancelled successfully")
            finally:
                self.processing_task = None

    async def abort(self, request_id: str):
        # 从 prefill_queue 和 decoding_queue 中移除 request_id
        self.logger.debug(f"abort: {request_id}")
        self.abort_queue.put_nowait(request_id)
