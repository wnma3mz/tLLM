# coding: utf-8
import asyncio
import math
import time
from typing import Dict, List, Tuple

import grpc

from tllm.commons.communicator import Communicator
from tllm.commons.convert import Convertor
from tllm.commons.manager import load_client_model
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.master_handler import PendingRequests
from tllm.schemas import MIX_TENSOR, SeqInput

grpc_options = [
    ("grpc.max_metadata_size", 32 * 1024 * 1024),
    ("grpc.max_send_message_length", 128 * 1024 * 1024),
    ("grpc.max_receive_message_length", 128 * 1024 * 1024),
]


class RPCManager:
    def __init__(self, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.grpc_options = grpc_options
        self.stub = None

    def update_url(self, url: str, pp_size: int):
        channel = grpc.aio.insecure_channel(url, options=self.grpc_options)
        self.stub = schemas_pb2_grpc.RPCServiceStub(channel)
        self.pp_size = pp_size

    async def rpc_forward(self, uuid, seq_len, hidden_states: schemas_pb2.BFloat16Tensor):
        forward_request = {"uuid": uuid, "seq_len": seq_len, "hidden_states": hidden_states}
        self.stub.Forward(schemas_pb2.ForwardRequest(**forward_request))

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        convertor = Convertor()
        hidden_states = convertor.serialize(hidden_states)
        # 发送完请求前，准备等待返回结果
        forward_future, status_future = self.pending_requests.add_request(
            "-".join(x for x in seq_input.uuid_list), self.pp_size
        )
        asyncio.create_task(self.rpc_forward(seq_input.uuid_list, seq_input.seq_len_list, hidden_states))
        await asyncio.sleep(0)
        try:
            output = await asyncio.wait_for(forward_future, timeout=100.0)  # 所有节点的总处理时间不超过 100s
        except asyncio.CancelledError:
            raise asyncio.CancelledError

        return convertor.deserialize(output), await asyncio.wait_for(status_future, timeout=100.0)

    async def rpc_image_forward(
        self,
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
        self.stub.ImageForward(schemas_pb2.ImageForwardRequest(**forward_request))

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
            "-".join(x for x in [request_id]), self.pp_size
        )
        asyncio.create_task(
            self.rpc_image_forward([request_id], hidden_states, text_embeddings, seq_len, height, width)
        )
        await asyncio.sleep(0)
        try:
            output = await asyncio.wait_for(forward_future, timeout=100.0)
        except asyncio.CancelledError:
            raise asyncio.CancelledError
        return convertor.deserialize(output), await asyncio.wait_for(status_future, timeout=100.0)


class LocalRPCManager:
    # 并不发生通信，直接调用模型
    def __init__(self, model_path: str):
        self.model = load_client_model(0, math.inf, Communicator(), model_path)

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        return output_hidden_states, [time.perf_counter() - s1]

    async def image_forward(
        self,
        hidden_states: MIX_TENSOR,
        text_embeddings: MIX_TENSOR,
        seq_len: int,
        height: int,
        width: int,
        request_id: str,
    ) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, text_embeddings, seq_len, height, width, [request_id])
        return output_hidden_states, [time.perf_counter() - s1]


class ClientRPCManager:
    def __init__(self, pp_size: int):
        self.stub_list = [None for _ in range(pp_size)]

    def update_url(self, url_list: List[str]):
        for i, url in enumerate(url_list):
            channel = grpc.aio.insecure_channel(url, options=grpc_options)
            self.stub_list[i] = schemas_pb2_grpc.RPCServiceStub(channel)

    async def set_config(self, idx: int, config: Dict) -> None:
        await self.stub_list[idx].SetConfig(schemas_pb2.SetConfigRequest(**config))

    async def health_check(self, idx: int) -> bool:
        try:
            await self.stub_list[idx].Health(schemas_pb2.Empty())
            return True
        except Exception as e:
            return False


class MasterRPCManager:
    # 向 Master 发送 gRPC 请求
    def __init__(self, grpc_options: List[Tuple[str, int]]):
        self.grpc_options = grpc_options
        self.master_stub = None

    def update_url(self, master_url: str, forward_url: str, pp_idx: int):
        master_channel = grpc.aio.insecure_channel(master_url, options=self.grpc_options)
        forward_channel = grpc.aio.insecure_channel(forward_url, options=self.grpc_options)
        self.master_stub = schemas_pb2_grpc.RPCServiceStub(master_channel)
        self.forward_stub = schemas_pb2_grpc.RPCServiceStub(forward_channel)
        self.pp_idx = pp_idx

    async def rpc_func(self, uuid, seq_len, hidden_states: schemas_pb2.BFloat16Tensor, cost_time: float):
        forward_request = {"uuid": uuid, "seq_len": seq_len, "hidden_states": hidden_states}
        status_request = {"uuid": uuid, "seq_len": seq_len, "pp_idx": self.pp_idx, "cost_time": cost_time}
        self.master_stub.Status(schemas_pb2.StatusRequest(**status_request))
        self.forward_stub.Forward(schemas_pb2.ForwardRequest(**forward_request))

    async def rpc_image_func(
        self,
        request: schemas_pb2.ImageForwardRequest,
        hidden_states: schemas_pb2.BFloat16Tensor,
        cost_time: float,
    ):
        # 最后一个 PP 不需要返回 text_embeddings 和 image_rotary_emb
        forward_request = {
            "uuid": request.uuid,
            "hidden_states": hidden_states,
            # "text_embeddings": request.text_embeddings,
            # "image_rotary_emb": request.image_rotary_emb,
        }
        status_request = {"uuid": request.uuid, "pp_idx": self.pp_idx, "cost_time": cost_time}
        await self.master_stub.Status(schemas_pb2.StatusRequest(**status_request))
        await self.forward_stub.ImageForward(schemas_pb2.ImageForwardRequest(**forward_request))
