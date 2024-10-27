import asyncio
import traceback
from typing import *

import torch

from tllm.models.protocol import SequenceRequestData
from tllm.models.utils import is_generate_end
from tllm.rpc.protocol import SeqInput


@torch.no_grad()
async def generate_utils(model, sequence_request_list: List[SequenceRequestData]) -> AsyncGenerator:
    """
    @params:
        sequence_request_list: List[Params]
            Params:
                input_ids: torch.Tensor

    """
    uuid_str_list, input_ids_list, seq_len_list = [], [], []
    for sequence_request in sequence_request_list:
        uuid_str_list.append(sequence_request.request_id)
        # 如果是 prefilling，则为 input_ids
        # 否则，为 output_ids[-1]
        # input_ids: bsz x seq_len
        if len(sequence_request.output_ids) == 0:
            input_ids_list.append(sequence_request.input_ids)
            seq_len_list.append(sequence_request.input_ids.shape[-1])
        else:
            input_ids_list.append(torch.tensor([sequence_request.output_ids[-1]]).unsqueeze(0))
            seq_len_list.append(1)

    input_ids = torch.cat(input_ids_list, dim=-1)
    input_embeds = model.embed_tokens(input_ids)

    seq_input = SeqInput(uuid_str_list=uuid_str_list, seq_len_list=seq_len_list)
    forward_result = model(input_embeds, seq_input)
    logits = forward_result.logits

    # 根据 seq 拆开，之后直接在 sampler 中处理
    seq_logits_list = torch.split(logits, seq_input.seq_len_list, dim=1)
    for seq_logits, sequence_request in zip(seq_logits_list, sequence_request_list):
        generate_ids = sequence_request.sampler.decode(seq_logits)
        generate_texts = [model.tok.decode([x]) for x in generate_ids]

        sequence_request.output_ids.append(generate_ids[0])
        sequence_request.generate_ids = generate_ids
        sequence_request.generate_texts = generate_texts

        end = is_generate_end(
            sequence_request.output_ids,
            eos_token_ids=model.eos_token_ids,
            max_new_tokens=sequence_request.sampling_params.get("max_new_tokens", 16),
        )
        if end.is_end:
            sequence_request.finish_reason_list = [end.finish_reason]
            sequence_request.is_stop = True
        else:
            sequence_request.output_text += generate_texts[0]  # 不添加 end text

        if len(sequence_request.output_ids) == 1:
            sequence_request.ttft_cost_time = forward_result.comm_cost_time_list
        else:
            sequence_request.tpot_cost_time = forward_result.comm_cost_time_list

    comm_cost_time_list = forward_result.comm_cost_time_list
    model.logger.debug(f"communication cost time: {",".join([f'{x:.4f}' for x in comm_cost_time_list])}")


class AsyncEngine:
    def __init__(self, logger, model):
        self.tok = model.tok
        self.model = model
        self.prefill_queue: asyncio.Queue = asyncio.Queue()
        self.decoding_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
        self.limit_size: int = 5  # 每次最多处理 5 个请求，prefill + decode
        self.logger = logger

    def preprocess(self, messages: List[Dict[str, Any]]) -> torch.Tensor:
        input_id_list = self.tok.preprocess(messages=messages).input_ids
        input_ids = torch.tensor(input_id_list).unsqueeze(0)
        return input_ids

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
