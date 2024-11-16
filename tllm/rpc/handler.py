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
from tllm.rpc.manager import RPCManager
from tllm.rpc.model_client import HandlerArgs, ModelClient
from tllm.schemas import SeqInput
from tllm.utils import setup_logger


class RPCHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: Communicator, model, rank: int, pp_rank: int, logger, ip_addr: str = "localhost"):
        self.comm = comm
        self.model = model
        self.rank = rank
        self.pp_rank = pp_rank
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

        # 在 load 模型的时候已经知道 PP IDX 和 总的 PP
        with open("./examples/config.json", "r") as f:
            config = json.load(f)
        self.pp_size = len(config["client"])
        if self.pp_rank == self.pp_size - 1:
            next_pp_url = config["master"]["url"]
        else:
            next_pp_url = config["client"][self.pp_rank + 1]["url"]

        self.manager = RPCManager(next_pp_url)
        self.status_manager = RPCManager(config["master"]["url"])  # Maybe Comm by WebSocket?

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

        await self.manager.rpc_forward(request.uuid, request.seq_len, output)
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
    model_client = ModelClient(logger=logger, args=handler_args)
    model_client.start()
    model = model_client.load_model(comm, args.model_path)

    rpc_servicer = RPCHandler(comm, model, comm.rank, args.pp_rank, logger, args.ip_addr)
    if comm.rank == 0:
        await rpc_servicer.start()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
