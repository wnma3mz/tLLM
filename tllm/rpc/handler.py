import argparse
import asyncio
from concurrent import futures
import json
import logging
import os
import time
from typing import *

import grpc
import torch

from tllm.commons.communicator import Communicator, SingleNodeCommunicator
from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.websocket_client import HandlerArgs, WebSocketClient
from tllm.schemas import SeqInput
from tllm.utils import setup_logger


class RPCManager:
    def __init__(self, url: str, grpc_options: List[Tuple[str, int]]):
        self.grpc_options = grpc_options
        channel = grpc.aio.insecure_channel(url, options=self.grpc_options)
        self.stub = schemas_pb2_grpc.RPCServiceStub(channel)

    def update_url(self, url: str):
        channel = grpc.aio.insecure_channel(url, options=self.grpc_options)
        self.stub = schemas_pb2_grpc.RPCServiceStub(channel)

    async def rpc_status(self, uuid, seq_len, pp_idx: int, cost_time: float):
        status_request = {"uuid": uuid, "seq_len": seq_len, "pp_idx": pp_idx, "cost_time": cost_time}
        self.stub.Status(schemas_pb2.StatusRequest(**status_request))

    async def rpc_forward(self, uuid, seq_len, hidden_states: schemas_pb2.BFloat16Tensor):
        forward_request = {"uuid": uuid, "seq_len": seq_len, "hidden_states": hidden_states}
        self.stub.Forward(schemas_pb2.ForwardRequest(**forward_request))


class RPCHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: Communicator, model, ws_client: WebSocketClient, logger, ip_addr: str = "localhost"):
        self.comm = comm
        self.model = model
        self.rank = comm.rank
        self.init_model_flag = False
        self.ip_addr = ip_addr
        self.prefix_log_str = f"IP: [{self.ip_addr}]"
        uuid_shape_list = [None, None]
        self.server = None
        self.logger = logger
        self.grpc_options = [
            ("grpc.max_metadata_size", 32 * 1024 * 1024),
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]

        forward_url, master_url = None, None
        while forward_url is None:
            # TODO 设置为事件通知
            # 如果发生变化，需要感知到
            message = ws_client.get_data()
            if message:
                self.pp_rank = message["pp_rank"]
                forward_url = message["forward_url"]
                master_url = message["master_url"]
            time.sleep(3)
        self.logger.info(f"[Master]: {master_url}")
        self.logger.info(f"[Forward]: {forward_url}")

        self.status_manager = RPCManager(master_url, self.grpc_options)
        self.forward_manager = RPCManager(forward_url, self.grpc_options)

        if self.rank == 0:
            pass
        else:
            while self.comm.world_size > 1:
                self.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.model.dtype)
                self.comm.broadcast(hidden_states)

                _ = self.model.forward(hidden_states, seq_input=seq_input)

    async def start(self):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=self.grpc_options)

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{args.port}")
        self.logger.info(f"Starting gRPC server on port {args.port}")
        await self.server.start()

        try:
            # 保持服务器运行
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            await self.stop()

    async def stop(self):
        if self.server:
            try:
                await self.server.stop(grace=5)
            except Exception as e:
                pass

    async def forward_func(self, request: schemas_pb2.ForwardRequest):
        s1 = time.perf_counter()
        hidden_states = deserialize_tensor(request.hidden_states)

        seq_input = SeqInput(uuid_list=list(request.uuid), seq_len_list=list(request.seq_len))
        self.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
        self.comm.broadcast(hidden_states)
        self.logger.debug(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        cost_time = time.perf_counter() - s1
        self.logger.debug(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = serialize_tensor(output_hidden_states)
        self.logger.debug(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        self.logger.debug("=" * 20)

        await self.forward_manager.rpc_forward(request.uuid, request.seq_len, output)
        await self.status_manager.rpc_status(request.uuid, request.seq_len, self.pp_rank, cost_time)

    async def Forward(
        self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: str
            seq_len: int
        """
        asyncio.create_task(self.forward_func(request))

        await asyncio.sleep(0)
        return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--master_url", type=str, default="ws://localhost:8000")
    parser.add_argument("--start_layer_idx", type=int, default=0, help="start layer idx")
    parser.add_argument("--end_layer_idx", type=int, default=11, help="end layer idx")
    parser.add_argument("--pp_rank", type=int, default=0, help="pp rank")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--ip_addr", type=str, default="localhost", help="提供给 server 连接的 ip")
    return parser.parse_args()


async def run(args):
    comm = SingleNodeCommunicator()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        comm = Communicator(is_torchrun=True)
    logger = setup_logger("handler_" + __name__, logging.DEBUG if args.is_debug else logging.INFO)

    handler_args = HandlerArgs(
        start_idx=args.start_layer_idx,
        end_idx=args.end_layer_idx,
        ip_addr=args.ip_addr,
        port=args.port,
        master_url=args.master_url,
    )
    ws_client = WebSocketClient(logger=logger, args=handler_args)
    ws_client.start()
    model = ws_client.load_model(comm, args.model_path)

    rpc_servicer = RPCHandler(comm, model, ws_client, logger, args.ip_addr)
    if comm.rank == 0:
        await rpc_servicer.start()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
