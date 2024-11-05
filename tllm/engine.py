import asyncio
import time
import traceback
from typing import *

from tllm.generate.llm_generator import LLMGenerator
from tllm.generate.token_utils import TokenizerUtils
from tllm.models.protocol import SequenceRequestData

conversations_dict = {}  # List[int] -> Tuple[str, int], TODO LRU 缓存


def post_process(data: SequenceRequestData):
    # TODO
    # 保存输入 + 输出
    # token_ids = data.input_ids + data.output_ids
    # conversations_dict[token_ids] = (data.history_request_id, len(token_ids)) if data.history_request_id else (data.request_id, len(token_ids))

    # 保存输入
    # token_ids = data.input_ids
    # conversations_dict[token_ids] = (data.history_request_id, len(token_ids)) if data.history_request_id else (data.request_id, len(token_ids))
    return


class MessageProcessor:
    # TODO async
    def __init__(self, tok: TokenizerUtils):
        self.tok = tok
        self.role_set = {"user", "system", "assistant"}

    def parse_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        new_messages = []
        for msg in messages:
            assert "role" in msg and "content" in msg, ValueError("role and content must be in message")
            if msg["role"] not in self.role_set:
                raise ValueError(f"role must be in {self.role_set}")
            new_messages.append({"role": msg["role"], "content": msg["content"]})
        return new_messages

    def preprocess(self, messages: List[Dict[str, str]]) -> List[int]:
        return self.tok.preprocess(messages=messages).input_ids

    def fetch_request_id(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], int]:
        # TODO
        # 根据 message 找到历史生成 request_id
        # 每轮对话以此向上查找
        # while messages:
        #     # QAQ 是否生成过
        #     input_ids = self.tok.preprocess(messages=messages).input_ids
        #     if input_ids in self.conversations_dict:
        #         return self.conversations_dict[input_ids]
        #     if len(messages) < 0:
        #         break
        #     # QA 是否生成过
        #     messages.pop()  # 去掉最新的Q
        #     input_ids = self.tok.preprocess(messages=messages, add_generation_prompt=False).input_ids
        #     if input_ids in self.conversations_dict:
        #         return self.conversations_dict[input_ids]
        return None, -1


class AsyncEngine:
    def __init__(self, logger, generator: LLMGenerator):
        self.generator = generator
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        self.limit_size: int = 5  # 每次最多处理 5 个请求，prefill + decode
        self.logger = logger
        self.abort_queue: asyncio.Queue = asyncio.Queue()

    async def fetch_data(self):
        aborting_request_ids = set()
        while not self.abort_queue.empty():
            request_id = await self.abort_queue.get()
            aborting_request_ids.add(request_id)

        async def aborting_filter(sequence_data) -> bool:
            if sequence_data.request_id in aborting_request_ids:
                self.logger.debug(f"aborting generate request_id: {sequence_data.request_id}")
                sequence_data.is_stop = True
                sequence_data.finish_reason_list = ["abort"]
                aborting_request_ids.remove(sequence_data.request_id)
                # async with sequence_data.condition:
                #     sequence_data.condition.notify()
                return True
            return False

        # prefill 队列和 decoding 队列的调度逻辑
        sequence_data_list = []

        # 优先从 decoding_queue 取数据
        while not self.decoding_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.decoding_queue.get()
            if await aborting_filter(sequence_data):
                continue
            sequence_data_list.append(sequence_data)

        # 从 prefill_queue 中取数据，直到达到限制
        while not self.prefill_queue.empty() and len(sequence_data_list) < self.limit_size:
            sequence_data = await self.prefill_queue.get()
            if await aborting_filter(sequence_data):
                continue
            sequence_data_list.append(sequence_data)

        return sequence_data_list

    async def _generate(self):
        while True:
            sequence_data_list: List[SequenceRequestData] = await self.fetch_data()
            if len(sequence_data_list) == 0:
                await asyncio.sleep(0.1)
                continue
            try:
                await self.generator.generate(sequence_data_list)

                for sequence_data in sequence_data_list:
                    if not sequence_data.is_stop:
                        await self.decoding_queue.put(sequence_data)
                    async with sequence_data.condition:
                        sequence_data.condition.notify()

                # await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing prefill_queue data: {str(e)}")
                traceback.print_exc()
            except BaseException as e:
                self.logger.error(f"BaseException Error processing prefill_queue data: {str(e)}")
                traceback.print_exc()
            finally:
                await asyncio.sleep(0.1)

    async def generate_stream(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
                    yield data.to_request_output()  # 流式返回数据的内容，可以控制
                # post_process(data)
                try:
                    self.logger.info(f"[request_id] {data.request_id}] ttft: {data.ttft_cost_time:.4f} s")
                    self.logger.info(
                        f"[request_id] {data.request_id}] tpot: {(len(data.output_ids) - 1) / (time.time() - data.decode_start_ts):.4f} token/s"
                    )
                except:
                    pass
        except asyncio.CancelledError:
            self.logger.debug(f"CancelledError: {data.request_id}")
            raise asyncio.CancelledError("CancelledError")
        except asyncio.TimeoutError:
            self.logger.debug(f"Processing timed out: {data.request_id}")
            raise asyncio.CancelledError("TimeoutError")
        except Exception as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownException")
        except BaseException as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownBaseException")

    async def generate(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
            return data.to_request_output()  # 返回最终的数据对象
        except asyncio.CancelledError:
            self.logger.debug(f"CancelledError: {data.request_id}")
            raise asyncio.CancelledError("CancelledError")
        except asyncio.TimeoutError:
            self.logger.debug(f"Processing timed out: {data.request_id}")
            raise asyncio.CancelledError("TimeoutError")
        except Exception as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownException")
        except BaseException as e:
            traceback.print_exc()
            raise asyncio.CancelledError("UnknownBaseException")

    async def start(self):
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._generate())

    async def stop(self):
        self.logger.info("Stopping processing sequence_data")
        if self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None

    async def abort(self, request_id: str):
        # 从 prefill_queue 和 decoding_queue 中移除 request_id
        self.logger.debug(f"abort: {request_id}")
        await self.abort_queue.put(request_id)
