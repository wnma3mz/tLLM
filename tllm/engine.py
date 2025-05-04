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

        async def aborting_filter(request_data) -> bool:
            if request_data.request_id in aborting_request_ids:
                self.logger.debug(f"aborting generate request_id: {request_data.request_id}")
                request_data.is_stop = True
                request_data.finish_reason_list = ["abort"]
                aborting_request_ids.remove(request_data.request_id)
                return True
            return False

        # prefill 队列和 decoding 队列的调度逻辑
        request_data_list = []

        # 优先从 decoding_queue 取数据
        while not self.decoding_queue.empty() and len(request_data_list) < self.limit_size:
            request_data = self.decoding_queue.get_nowait()
            if await aborting_filter(request_data):
                continue
            request_data_list.append(request_data)

        # 从 prefill_queue 中取数据，直到达到限制
        while not self.prefill_queue.empty() and len(request_data_list) < self.limit_size:
            request_data = self.prefill_queue.get_nowait()
            if await aborting_filter(request_data):
                continue
            request_data_list.append(request_data)

        return request_data_list

    async def _generate(self):
        while True:
            request_data_list: List[SequenceRequestData] = await self.fetch_data()
            if len(request_data_list) == 0:
                try:
                    await self.queue_not_empty.wait()
                except Exception as e:
                    self.logger.debug(f"exception: {str(e)}")
                    await asyncio.sleep(0.01)
                continue
            try:
                await self.generator.generate(request_data_list)

                for request_data in request_data_list:
                    if not request_data.is_stop:
                        self.decoding_queue.put_nowait(request_data)

            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                for request_data in request_data_list:
                    request_data.is_normal_process = False
                self.logger.error(f"{type(e).__name__}")
                traceback.print_exc()
            except Exception as e:
                self.logger.error(f"Error processing prefill_queue data: {repr(e)}")
                self.logger.error(f"Error request_id: {','.join(x.request_id for x in request_data_list)}")
                traceback.print_exc()
            except BaseException as e:
                self.logger.error(f"BaseException Error processing prefill_queue data: {str(e)}")
                traceback.print_exc()
            finally:
                for request_data in request_data_list:
                    async with request_data.condition:
                        request_data.condition.notify()
                if self.prefill_queue.empty():
                    self.queue_not_empty.clear()
                await asyncio.sleep(self.sleep_time)

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
        except asyncio.CancelledError:
            self.logger.debug(f"CancelledError: {request_data.request_id}")
            raise asyncio.CancelledError("CancelledError")
        except asyncio.TimeoutError:
            self.logger.debug(f"Processing timed out: {request_data.request_id}")
            raise asyncio.CancelledError("TimeoutError")
        except Exception as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownException")
        except BaseException as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownBaseException")

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
