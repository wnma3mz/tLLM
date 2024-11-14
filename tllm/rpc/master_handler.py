# coding: utf-8
import asyncio
from concurrent import futures
from typing import *
from typing import Any, Dict

import grpc

from tllm.rpc import schemas_pb2, schemas_pb2_grpc


class StatusTracker:
    def __init__(self, target_count: int):
        self.target_count = target_count
        self.current_count = 0
        self.is_completed = False
        self.future = asyncio.Future()
        self.pp_cost_time = [0 for _ in range(target_count)]

    def update(self, count: int, result: Tuple[int, float]):
        self.current_count = count
        self.pp_cost_time[result[0]] = result[1]
        if self.current_count >= self.target_count:
            self.is_completed = True
            self.future.set_result(self.pp_cost_time)


class PendingRequests:
    """管理待处理的请求"""

    def __init__(self):
        self._forward_requests: Dict[str, asyncio.Future] = {}
        self._status_requests: Dict[str, StatusTracker] = {}

    def add_request(self, trace_id: str, pp_size: int) -> Tuple[asyncio.Future, asyncio.Future]:
        forward_future = asyncio.Future()
        self._forward_requests[trace_id] = forward_future
        status_tracker = StatusTracker(pp_size)
        self._status_requests[trace_id] = status_tracker
        return forward_future, status_tracker.future

    def complete_forward_request(self, trace_id: str, result: Any) -> bool:
        if trace_id in self._forward_requests:
            future = self._forward_requests[trace_id]
            if not future.done():
                future.set_result(result)
                del self._forward_requests[trace_id]
                return True
        return False

    def complete_status_request(self, trace_id: str, result: Any) -> bool:
        if trace_id in self._status_requests:
            tracker = self._status_requests[trace_id]
            tracker.update(tracker.current_count + 1, result)
            return tracker.is_completed
        return False

    def fail_forward_request(self, trace_id: str, error: Exception):
        if trace_id in self._forward_requests:
            future = self._forward_requests[trace_id]
            if not future.done():
                future.set_exception(error)
            del self._forward_requests[trace_id]

    def fail_status_request(self, trace_id: str, error: Exception):
        if trace_id in self._status_requests:
            tracker = self._status_requests[trace_id]
            if not tracker.future.done():
                tracker.future.set_exception(error)
            del self._status_requests[trace_id]


class MasterHandler(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, logger, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.logger = logger

    async def start(self, port: int):
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_metadata_size", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 128 * 1024 * 1024),
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ],
        )

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.logger.info(f"Starting Master gRPC server on port {port}")
        await self.server.start()

    async def stop(self):
        if self.server:
            try:
                await self.server.stop(grace=5)
                await self.server.wait_for_termination()
            except Exception as e:
                pass

    async def Forward(
        self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """处理从最后一个节点返回的结果"""
        request_id = "-".join(x for x in list(request.uuid))
        self.logger.debug(f"Received result request id: {request_id}")

        try:
            self.pending_requests.complete_forward_request(request_id, request.hidden_states)
        except Exception as e:
            self.logger.debug("error")
        except BaseException as e:
            self.logger.debug("base error")
        finally:
            return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    async def Status(
        self, request: schemas_pb2.StatusRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.StatusResponse:
        request_id = "-".join(x for x in list(request.uuid))

        try:
            self.pending_requests.complete_status_request(request_id, (int(request.pp_idx), request.cost_time))
        except Exception as e:
            self.logger.debug("error")
        except BaseException as e:
            self.logger.debug("base error")
        finally:
            return schemas_pb2.StatusResponse(msg="Successful", status=200)
