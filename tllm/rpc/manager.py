# coding: utf-8
import asyncio
import time
from typing import *

import grpc

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.master_handler import PendingRequests
from tllm.rpc.websocket_client import HandlerArgs, WebSocketClient
from tllm.schemas import MIX_TENSOR, SeqInput


class RPCManager:
    def __init__(self, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.grpc_options = [
            ("grpc.max_metadata_size", 32 * 1024 * 1024),
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]
        self.stub = None

    def update_url(self, url: str, pp_size: int):
        channel = grpc.aio.insecure_channel(url, options=self.grpc_options)
        self.stub = schemas_pb2_grpc.RPCServiceStub(channel)
        self.pp_size = pp_size

    async def rpc_forward(self, uuid, seq_len, hidden_states: schemas_pb2.BFloat16Tensor):
        forward_request = {"uuid": uuid, "seq_len": seq_len, "hidden_states": hidden_states}
        self.stub.Forward(schemas_pb2.ForwardRequest(**forward_request))

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        hidden_states = serialize_tensor(hidden_states)
        # 发送完请求前，准备等待返回结果
        forward_future, status_future = self.pending_requests.add_request(
            "-".join(x for x in seq_input.uuid_list), self.pp_size
        )
        asyncio.create_task(self.rpc_forward(seq_input.uuid_list, seq_input.seq_len_list, hidden_states))
        await asyncio.sleep(0)
        output = await asyncio.wait_for(forward_future, timeout=100.0)  # 所有节点的总处理时间不超过 100s

        return deserialize_tensor(output), await asyncio.wait_for(status_future, timeout=100.0)


class LocalRPCManager:
    # 并不发生通信，直接调用模型
    def __init__(self, logger, model_path: str, num_hidden_layers: int):
        handler_args = HandlerArgs(
            ip_addr="localhost",
            port=-1,
            start_idx=0,
            end_idx=num_hidden_layers,
            master_url="localhost",
        )
        ws_client = WebSocketClient(logger=logger, args=handler_args)
        self.model = ws_client.load_model(SingleNodeCommunicator(), model_path)

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        return output_hidden_states, [time.perf_counter() - s1]


if __name__ == "__main__":
    # for test
    server = RPCManager("localhost:50051")
    server.post_sync(0, "/forward", {"uuid": "123", "hidden_states": [[1.0, 2.0, 3.0]]})
