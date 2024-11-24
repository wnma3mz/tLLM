import argparse
import asyncio
from concurrent import futures
import logging
import os
import time
from typing import List, Tuple
import uuid

import aiohttp
import grpc
import torch

from tllm.commons.communicator import Communicator, SingleNodeCommunicator
from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.models.manager import ModelManager
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.schemas import InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse, SeqInput
from tllm.utils import setup_logger


async def register_client(url: str, request_data: RegisterClientRequest):
    url = f"{url}/register_client"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request_data.dict(), timeout=3) as response:
            return RegisterClientResponse(**await response.json())


async def init_model(url: str, request_data: InitModelRequest):
    url = f"{url}/init_model"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request_data.dict(), timeout=3) as response:
            return InitModelResponse(**await response.json())


class RPCManager:
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


class RPCHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: Communicator, logger, master_url: str, ip_addr: str = "localhost", port: int = 50051):
        self.comm = comm
        self.rank = comm.rank
        self.ip_addr = ip_addr
        self.port = port
        self.prefix_log_str = f"IP: [{self.ip_addr}]"
        self.client_id = f"{str(uuid.uuid4())[:8]}"

        uuid_shape_list = [None, None]
        self.server = None
        self.logger = logger
        self.master_url = master_url

        self.grpc_options = [
            ("grpc.max_metadata_size", 32 * 1024 * 1024),
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ]

        self.init_model_info = None
        self.manager = RPCManager(self.grpc_options)

        if self.rank == 0:
            pass
        else:
            while self.comm.world_size > 1:
                self.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.model.dtype)
                self.comm.broadcast(hidden_states)

                _ = self.model.forward(hidden_states, seq_input=seq_input)

    async def load_model(self, model: str, start_idx: int, end_idx: int):
        model_manager = ModelManager(start_idx, end_idx)
        self.model = model_manager.load_model(self.comm, model)
        self.logger.info(f"Model loaded")

    async def connect(self):
        """定期发送连接请求的协程"""
        while self.is_running:
            try:
                if not self.init_model_info:
                    register_request = RegisterClientRequest(
                        client_id=self.client_id, host=f"{self.ip_addr}:{self.port}"
                    )
                    response: RegisterClientResponse = await register_client(self.master_url, register_request)

                    await self.load_model(response.model, response.start_idx, response.end_idx)

                    self.init_model_info = {
                        "pp_rank": response.pp_rank,
                        "start_idx": response.start_idx,
                        "end_idx": response.end_idx,
                    }
                    init_request = InitModelRequest(client_id=self.client_id, **self.init_model_info)
                    response = await init_model(self.master_url, init_request)
                    self.logger.info(f"Connection successful: {response}")
                else:
                    register_request = RegisterClientRequest(
                        client_id=self.client_id,
                        host=f"{self.ip_addr}:{self.port}",
                        pp_rank=self.init_model_info["pp_rank"],
                        start_idx=self.init_model_info["start_idx"],
                        end_idx=self.init_model_info["end_idx"],
                    )
                    response: RegisterClientResponse = await register_client(self.master_url, register_request)

                self.is_running = False
            except grpc.RpcError as e:
                self.logger.error(f"Connection failed: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")

            # 等待一段时间后再次尝试连接
            await asyncio.sleep(10)

    async def start(self):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=self.grpc_options)

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.logger.info(f"Starting gRPC server on port {self.port}")
        await self.server.start()

        self.is_running = True
        connection_task = asyncio.create_task(self.connect())

        try:
            # 保持服务器运行
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            self.is_running = False
            connection_task.cancel()
            try:
                await connection_task
            except asyncio.CancelledError:
                pass
            await self.stop()

    async def stop(self):
        self.is_running = False
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

        await self.manager.rpc_func(request.uuid, request.seq_len, output, cost_time)

    async def SetConfig(
        self, request: schemas_pb2.SetConfigRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.SetConfigResponse:
        self.manager.update_url(request.master_url, request.forward_url, request.pp_rank)
        return schemas_pb2.SetConfigResponse(msg="SetConfig Completed", status=200)

    async def Forward(
        self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: str
            seq_len: int
        """
        if not hasattr(self, "model"):
            return schemas_pb2.ForwardResponse(msg="Model not initialized", status=400)
        if hasattr(self.manager, "master_stub") is None:
            return schemas_pb2.ForwardResponse(msg="Manager not initialized", status=400)
        asyncio.create_task(self.forward_func(request))

        await asyncio.sleep(0)
        return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--master_url", type=str, default="http://localhost:8000", help="master 的地址")
    parser.add_argument("--ip_addr", type=str, default="localhost", help="提供给 master 连接的 ip, 如 localhost")
    parser.add_argument("--is_debug", action="store_true")
    return parser.parse_args()


async def run(args):
    comm = SingleNodeCommunicator()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        comm = Communicator(is_torchrun=True)
    logger_name = "handler"
    logger = setup_logger(logger_name, logging.DEBUG if args.is_debug else logging.INFO)

    rpc_servicer = RPCHandler(comm, logger, args.master_url, args.ip_addr, args.port)
    if comm.rank == 0:
        await rpc_servicer.start()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
