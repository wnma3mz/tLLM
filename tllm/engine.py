import asyncio
import time
import traceback
from typing import *

import numpy as np
import torch

from tllm.generate.token_utils import TokenizerUtils
from tllm.models.protocol import SeqInput, SequenceRequestData
from tllm.models.utils import is_generate_end


@torch.no_grad()
async def generate_utils(model, sequence_request_list: List[SequenceRequestData]) -> AsyncGenerator:
    """
    @params:
        sequence_request_list: List[SequenceRequestData]
            Params:
                input_ids: torch.Tensor

    """
    uuid_list, input_ids_list, seq_len_list = [], [], []
    for sequence_request in sequence_request_list:
        uuid_list.append(sequence_request.request_id)
        # 如果是 prefilling，则为 input_ids
        # 否则，为 output_ids[-1]
        # input_ids: bsz x seq_len
        if sequence_request.is_prefill:
            if sequence_request.history_request_id:
                uuid_list[-1] = sequence_request.history_request_id
            input_ids_list.append(np.array(sequence_request.input_ids).reshape(1, -1))
            seq_len_list.append(sequence_request.q_len)
        else:
            input_ids_list.append(np.array(sequence_request.output_ids[-1]).reshape(1, -1))
            seq_len_list.append(1)

    input_ids = np.concatenate(input_ids_list, axis=-1)
    input_embeds = model.get_input_embeddings(input_ids)

    seq_input = SeqInput(uuid_list=uuid_list, seq_len_list=seq_len_list)
    s1 = time.time()
    forward_result = model(input_embeds, seq_input)
    generate_time = time.time() - s1
    logits = forward_result.logits

    s1 = time.time()
    # 根据 seq 拆开，之后直接在 sampler 中处理
    seq_logits_list = torch.split(logits, [1 for _ in seq_input.seq_len_list], dim=1)
    for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
        generate_ids = sequence_request.sampler.sampling(seq_logits, sequence_request.sampling_params)
        generate_texts = sequence_request.sampler.decode(generate_ids)

        sequence_request.output_ids.append(generate_ids[0])

        end = is_generate_end(
            sequence_request.output_ids,
            eos_token_ids=model.eos_token_ids,
            max_tokens=sequence_request.sampling_params.max_tokens,
        )
        if end.is_end:
            sequence_request.finish_reason_list = [end.finish_reason]
            sequence_request.is_stop = True
        else:
            sequence_request.generate_text = generate_texts[0]
            sequence_request.output_text += generate_texts[0]  # 不添加 end text

        if sequence_request.is_prefill:
            sequence_request.ttft_cost_time = generate_time
            sequence_request.decode_start_ts = time.time()
            sequence_request.is_prefill = False

    comm_cost_time_list = forward_result.comm_cost_time_list
    comm_cost_time_str = ",".join([f"{x:.4f}" for x in comm_cost_time_list])
    sum_comm = sum(comm_cost_time_list)
    fraction = sum_comm / (sum_comm + sum(forward_result.calc_cost_time_list))
    model.logger.debug(f"communication cost time: {comm_cost_time_str}s({fraction*100:.1f}%)")
    model.logger.debug(f"decoder cost time: {time.time() - s1:.4f}s")
    model.logger.debug(f"forward cost time: {sum(forward_result.calc_cost_time_list):.4f}s")
    model.logger.debug("=" * 5)


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
    def __init__(self, logger, model):
        self.model = model
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
                async with sequence_data.condition:
                    sequence_data.condition.notify()
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
                await generate_utils(self.model, sequence_data_list)

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
                    self.logger.debug(f"[request_id] {data.request_id}] ttft: {data.ttft_cost_time:.4f} s")
                    self.logger.debug(
                        f"[request_id] {data.request_id}] tpot: {(len(data.output_ids) - 1) / (time.time() - data.decode_start_ts):.4f} token/s"
                    )
                except:
                    pass
        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")
        except Exception as e:
            traceback.print_exc()
        except BaseException as e:
            traceback.print_exc()

    async def generate(self, data: SequenceRequestData):
        await self.prefill_queue.put(data)

        try:
            async with data.condition:
                while not data.is_stop:
                    await asyncio.wait_for(data.condition.wait(), data.timeout)
            return data.to_request_output()  # 返回最终的数据对象
        except asyncio.TimeoutError:
            raise TimeoutError("Processing timed out")

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
