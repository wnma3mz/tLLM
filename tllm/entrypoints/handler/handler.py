import asyncio
from concurrent import futures
import time
from typing import List
import uuid

import grpc

from tllm import CLIENT_SOCKET_PATH, GRPC_OPTIONS, MASTER_SOCKET_PATH
from tllm.commons.communicator import BaseCommunicator, Communicator
from tllm.commons.convert import Convertor
from tllm.entrypoints.utils import parse_handler_args, update_handler_args
from tllm.network.http_client import HTTPClient
from tllm.network.manager import MasterRPCManager
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.schemas import SeqInput
from tllm.singleton_logger import SingletonLogger


class RPCHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: BaseCommunicator, logger, master_url: str):
        self.comm = comm
        self.client_id = f"{str(uuid.uuid4())[:8]}"

        uuid_shape_list = [None, None]
        self.server = None
        self.logger = logger

        self.grpc_options = GRPC_OPTIONS

        self.manager = MasterRPCManager(self.grpc_options)
        self.http_client = HTTPClient(master_url, comm, logger)

        if comm.rank == 0:
            pass
        else:
            import torch

            while self.comm.world_size > 1:
                self.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.http_client.model.dtype)
                self.comm.broadcast(hidden_states)

                _ = self.http_client.model.forward(hidden_states, seq_input=seq_input)

    async def start(self, ip_addr_list: List[str], port: int = 50051):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=self.grpc_options)

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.server.add_insecure_port(f"unix://{CLIENT_SOCKET_PATH}")
        self.logger.info(f"Starting gRPC server on [::]:{port}")
        await self.server.start()

        self.http_client.is_running = True
        connection_task = asyncio.create_task(self.http_client.connect(self.client_id, ip_addr_list, port))
        ping_task = asyncio.create_task(self.http_client.maintain_connection(self.client_id, ip_addr_list, port))

        try:
            await self.server.wait_for_termination()
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.http_client.is_running = False
            connection_task.cancel()
            ping_task.cancel()
            try:
                await asyncio.gather(connection_task, ping_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            await self.stop()

    async def stop(self):
        self.http_client.is_running = False
        if self.server is not None:
            try:
                await self.server.stop(grace=5)
            except (Exception, asyncio.CancelledError):
                pass

    async def forward_func(self, request: schemas_pb2.ForwardRequest):
        s1 = time.perf_counter()
        convertor = Convertor()
        hidden_states = convertor.deserialize(request.hidden_states)

        seq_input = SeqInput(uuid_list=list(request.uuid), seq_len_list=list(request.seq_len))
        self.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
        self.comm.broadcast(hidden_states)
        self.logger.debug(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.http_client.model(hidden_states, seq_input)
        cost_time = time.perf_counter() - s1
        self.logger.debug(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = convertor.serialize(output_hidden_states)
        self.logger.debug(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        self.logger.debug("=" * 20)

        await self.manager.rpc_func(request.uuid, request.seq_len, output, cost_time)

    async def SetConfig(
        self, request: schemas_pb2.SetConfigRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.SetConfigResponse:
        self.logger.debug(f"forward_url: {request.forward_url}")
        self.manager.update_url(request.master_url, request.forward_url, request.pp_rank)
        return schemas_pb2.SetConfigResponse(msg="SetConfig Completed", status=200)

    async def Forward(
        self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: List[str]
            seq_len: List[int]
        """
        if hasattr(self.http_client, "model") is None:
            return schemas_pb2.ForwardResponse(msg="Model not initialized", status=400)
        if hasattr(self.manager, "master_stub") is None:
            return schemas_pb2.ForwardResponse(msg="Manager not initialized", status=400)
        asyncio.create_task(self.forward_func(request))

        await asyncio.sleep(0)
        return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    async def image_forward_func(self, request: schemas_pb2.ImageForwardRequest):
        s1 = time.perf_counter()
        import mlx.core as mx
        import numpy as np

        convertor = Convertor(mx.float32, np.float32, mx.float32)

        hidden_states = convertor.deserialize(request.hidden_states)
        text_embeddings = convertor.deserialize(request.text_embeddings)

        self.logger.debug(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.http_client.model(
            hidden_states, text_embeddings, request.seq_len, request.height, request.width, request.uuid
        )
        cost_time = time.perf_counter() - s1
        self.logger.debug(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = convertor.serialize(output_hidden_states)
        self.logger.debug(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        self.logger.debug("=" * 20)

        await self.manager.rpc_image_func(request, output, cost_time)

    async def ImageForward(
        self, request: schemas_pb2.ImageForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        if not hasattr(self.http_client, "model") and self.http_client is None:
            return schemas_pb2.ForwardResponse(msg="Model not initialized", status=400)
        if hasattr(self.manager, "master_stub") is None:
            return schemas_pb2.ForwardResponse(msg="Manager not initialized", status=400)
        asyncio.create_task(self.image_forward_func(request))

        await asyncio.sleep(0)
        return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    async def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)


async def run(args):
    SingletonLogger.set_level("DEBUG" if args.is_debug else "INFO")
    args, ip_addr_list = update_handler_args(args)
    comm = Communicator()

    logger = SingletonLogger.setup_handler_logger(f"handler-{args.grpc_port}")

    rpc_servicer = RPCHandler(comm, logger, args.master_addr)
    # if comm.rank == 0:
    try:
        await rpc_servicer.start(ip_addr_list, args.grpc_port)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        await rpc_servicer.stop()
        raise
    finally:
        await rpc_servicer.stop()


def main():
    args = parse_handler_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
