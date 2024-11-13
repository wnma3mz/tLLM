# coding: utf-8
import asyncio
from typing import *
from typing import Any, Dict

import grpc

from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from concurrent import futures


class PendingRequests:
    """管理待处理的请求"""

    def __init__(self):
        self._requests: Dict[str, asyncio.Future] = {}

    def add_request(self, trace_id: str) -> asyncio.Future:
        future = asyncio.Future()
        self._requests[trace_id] = future
        return future

    def complete_request(self, trace_id: str, result: Any) -> bool:
        if trace_id in self._requests:
            future = self._requests[trace_id]
            if not future.done():
                future.set_result(result)
                del self._requests[trace_id]
                return True
        return False

    def fail_request(self, trace_id: str, error: Exception):
        if trace_id in self._requests:
            future = self._requests[trace_id]
            if not future.done():
                future.set_exception(error)
            del self._requests[trace_id]


class MasterHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, logger, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.logger = logger

    def start(self, port: int):
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_metadata_size", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 128 * 1024 * 1024),
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ],
        )

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.logger.info(f"Starting gRPC server on port {port}")
        self.server.start()

    def Forward(self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext):
        """处理从最后一个节点返回的结果"""
        request_id = request.request_id
        self.logger.info(f"Received result from Service for trace_id: {request_id}")

        if self.pending_requests.complete_request(request_id, request):
            return schemas_pb2.ForwardResponse(
                msg="Forward pass completed",
                status=200,
                cost_time=0.0,
            )
        else:
            self.logger.debug("error")
            # await context.abort(grpc.StatusCode.NOT_FOUND, f"No pending request for trace_id: {trace_id}")
