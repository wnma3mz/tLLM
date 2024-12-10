# coding: utf-8
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union

import grpc

from tllm import GRPC_OPTIONS
from tllm.commons.convert import Convertor
from tllm.entrypoints.handler.master_handler import PendingRequests
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.schemas import MIX_TENSOR, SeqInput


async def rpc_image_forward(
    stub,
    request_id: str,
    hidden_states: schemas_pb2.BFloat16Tensor,
    text_embeddings: schemas_pb2.BFloat16Tensor,
    seq_len: int,
    height: int,
    width: int,
):
    forward_request = {
        "uuid": request_id,
        "hidden_states": hidden_states,
        "text_embeddings": text_embeddings,
        "seq_len": seq_len,
        "height": height,
        "width": width,
    }
    stub.ImageForward(schemas_pb2.ImageForwardRequest(**forward_request))


async def rpc_forward(stub, uuid, seq_len, hidden_states: schemas_pb2.BFloat16Tensor):
    forward_request = {"uuid": uuid, "seq_len": seq_len, "hidden_states": hidden_states}
    stub.Forward(schemas_pb2.ForwardRequest(**forward_request))


async def rpc_set_config(stub, config: Dict):
    await stub.SetConfig(schemas_pb2.SetConfigRequest(**config))


async def rpc_health_check(stub):
    await stub.Health(schemas_pb2.Empty())


class RPCManager:
    def __init__(self, client_size: int, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.grpc_options = GRPC_OPTIONS
        self.client_size = client_size
        self.stub_list = [None for _ in range(client_size)]

        self.task: Optional[asyncio.Task] = None
        self.last_check_result: List[int] = []
        self.last_check_time: Optional[float] = None

    def update_url(self, url_list: List[str]):
        for i, url in enumerate(url_list):
            channel = grpc.aio.insecure_channel(url, options=self.grpc_options)
            self.stub_list[i] = schemas_pb2_grpc.RPCServiceStub(channel)

    async def send_config(self, master_url: str, host_list: List[str]):
        assert len(host_list) == self.client_size

        async def set_single_config(i: int) -> None:
            url = master_url if i == self.client_size - 1 else host_list[i + 1]
            await rpc_set_config(self.stub_list[i], {"forward_url": url, "master_url": master_url, "pp_rank": i})

        tasks = [set_single_config(i) for i in range(self.client_size)]
        await asyncio.gather(*tasks)

    async def health_check(self) -> Tuple[int]:
        async def check_single_client(index: int) -> Tuple[int, bool]:
            try:
                await rpc_health_check(self.stub_list[index])
                return (index, True)
            except Exception as e:
                return (index, False)

        tasks = [check_single_client(i) for i in range(self.client_size)]

        # 等待所有任务完成，返回结果列表
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # 检查结果，如果有健康检查失败，返回对应的索引
        self.last_check_time = time.time()
        self.last_check_result = [index for index, is_healthy in results if not is_healthy]
        return self.last_check_result

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        convertor = Convertor()
        hidden_states = convertor.serialize(hidden_states)
        # 发送完请求前，准备等待返回结果
        forward_future, status_future = self.pending_requests.add_request(
            "-".join(x for x in seq_input.uuid_list), self.client_size
        )
        asyncio.create_task(rpc_forward(self.stub_list[0], seq_input.uuid_list, seq_input.seq_len_list, hidden_states))
        await asyncio.sleep(0)
        try:
            output = await asyncio.wait_for(forward_future, timeout=100.0)  # 所有节点的总处理时间不超过 100s
        except asyncio.CancelledError:
            raise asyncio.CancelledError

        return convertor.deserialize(output), await asyncio.wait_for(status_future, timeout=100.0)

    async def image_forward(
        self,
        hidden_states: MIX_TENSOR,
        text_embeddings: MIX_TENSOR,
        seq_len: int,
        height: int,
        width: int,
        request_id: str,
    ) -> Tuple[MIX_TENSOR, List[float]]:
        import mlx.core as mx
        import numpy as np

        convertor = Convertor(mx.float32, np.float32, mx.float32)

        hidden_states = convertor.serialize(hidden_states)
        text_embeddings = convertor.serialize(text_embeddings)
        forward_future, status_future = self.pending_requests.add_request(
            "-".join(x for x in [request_id]), self.client_size
        )
        asyncio.create_task(
            rpc_image_forward(self.stub_list[0], [request_id], hidden_states, text_embeddings, seq_len, height, width)
        )
        await asyncio.sleep(0)
        try:
            output = await asyncio.wait_for(forward_future, timeout=100.0)
        except asyncio.CancelledError:
            raise asyncio.CancelledError
        return convertor.deserialize(output), await asyncio.wait_for(status_future, timeout=100.0)

    async def start_health_check(self, interval: float = 10):
        if self.task and not self.task.done():
            return

        async def check_loop():
            while True:
                result = await self.health_check()
                if len(result) > 0:
                    break
                await asyncio.sleep(interval)

        self.task = asyncio.create_task(check_loop())

    def stop_health_check(self):
        self.last_check_time = None
        self.last_check_result = []
        if self.task and not self.task.done():
            self.task.cancel()

    async def get_status(self) -> Dict[str, Union[bool, float, List[int]]]:
        return {
            "last_check_time": self.last_check_time,
            "last_check_result": self.last_check_result,
            "is_running": bool(self.task and not self.task.done()),
        }
