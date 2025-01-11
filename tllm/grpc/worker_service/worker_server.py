import asyncio
from concurrent import futures
import time
from typing import List
import uuid

import grpc

from tllm import CLIENT_SOCKET_PATH, GRPC_OPTIONS
from tllm.commons.convert import Convertor
from tllm.commons.manager import load_client_model
from tllm.commons.tp_communicator import BaseCommunicator, Communicator
from tllm.entrypoints.utils import parse_handler_args, update_handler_args
from tllm.grpc.proto import schemas_pb2, schemas_pb2_grpc
from tllm.grpc.worker_service.http_client import HTTPClient
from tllm.grpc.worker_service.master_manager import MasterRPCManager
from tllm.schemas import SeqInput
from tllm.singleton_logger import SingletonLogger


class WorkerServer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: BaseCommunicator, logger, master_url: str, client_id: str):
        self.comm = comm
        self.client_id = client_id

        self.server = None
        self.model = None
        self.logger = logger

        self.grpc_options = GRPC_OPTIONS

        self.master_rpc_manager = MasterRPCManager(self.grpc_options)
        self.http_client = HTTPClient(master_url, comm, logger)

    def load_model_func(self, model_path: str, start_idx: int, end_idx: int):
        self.model = load_client_model(start_idx, end_idx, self.comm, model_path)

    async def start(self, ip_addr_list: List[str], port: int = 50051):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=self.grpc_options)

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        # self.server.add_insecure_port(f"unix://{CLIENT_SOCKET_PATH}_{self.comm.rank}")
        self.logger.info(f"Starting gRPC server on [::]:{port}")
        await self.server.start()

        self.http_client.is_running = True
        ping_task = asyncio.create_task(
            self.http_client.maintain_connection(
                self.client_id,
                ip_addr_list,
                port,
                self.load_model_func,
                self.comm.rank if self.comm.world_size > 1 else -1,
            )
        )

        try:
            await self.server.wait_for_termination()
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.http_client.is_running = False
            ping_task.cancel()
            try:
                await asyncio.gather(ping_task, return_exceptions=True)
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

        self.comm.debug_rank0(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        cost_time = time.perf_counter() - s1
        self.comm.debug_rank0(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = convertor.serialize(output_hidden_states)
        self.comm.debug_rank0(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        self.comm.debug_rank0("=" * 20)

        if self.comm.is_rank0():
            await self.master_rpc_manager.rpc_func(request.uuid, request.seq_len, output, cost_time)

    async def SetConfig(
        self, request: schemas_pb2.SetConfigRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.SetConfigResponse:
        self.comm.debug_rank0(f"forward_url: {request.forward_url}")
        self.master_rpc_manager.update_url(request.master_url, request.forward_url, request.pp_rank)
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
        if self.model is None:
            return schemas_pb2.ForwardResponse(msg="Model not initialized", status=500)
        if hasattr(self.master_rpc_manager, "master_stub") is None:
            return schemas_pb2.ForwardResponse(msg="Manager not initialized", status=500)
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

        self.comm.debug_rank0(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.model(
            hidden_states, text_embeddings, request.seq_len, request.height, request.width, request.uuid
        )
        cost_time = time.perf_counter() - s1
        self.comm.debug_rank0(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = convertor.serialize(output_hidden_states)
        self.comm.debug_rank0(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        self.comm.debug_rank0("=" * 20)

        await self.master_rpc_manager.rpc_image_func(request, output, cost_time)

    async def ImageForward(
        self, request: schemas_pb2.ImageForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        if self.model is None:
            return schemas_pb2.ForwardResponse(msg="Model not initialized", status=500)
        if hasattr(self.master_rpc_manager, "master_stub") is None:
            return schemas_pb2.ForwardResponse(msg="Manager not initialized", status=500)
        asyncio.create_task(self.image_forward_func(request))

        await asyncio.sleep(0)
        return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    async def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)


async def run(args):
    SingletonLogger.set_level("DEBUG" if args.is_debug else "INFO")
    args, ip_addr_list = update_handler_args(args)

    logger = SingletonLogger.setup_handler_logger(f"handler-{args.grpc_port}")
    # comm = Communicator(logger)
    comm_ip_list = ["192.168.124.30", "192.168.124.5"]
    comm_port_list = [50051, 50051]
    comm = Communicator(logger, 2, 0, comm_ip_list, comm_port_list)

    logger.info(f"[MLXCommunicator] Rank: {comm.rank}; World Size: {comm.world_size}")

    client_id = f"test-{str(uuid.uuid4())[:8]}-{comm.rank}"
    rpc_servicer = WorkerServer(comm, logger, args.master_addr, client_id)
    logger.info("args: %s", args)
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
