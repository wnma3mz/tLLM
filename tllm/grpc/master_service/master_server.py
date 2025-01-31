# coding: utf-8
import asyncio
from concurrent import futures

import grpc

from tllm import GRPC_OPTIONS, MASTER_SOCKET_PATH
from tllm.grpc.master_service.pending_requests import PendingRequests
from tllm.grpc.proto import schemas_pb2, schemas_pb2_grpc
from tllm.singleton_logger import SingletonLogger


class MasterServer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, pending_requests: PendingRequests):
        self.pending_requests = pending_requests
        self.logger = SingletonLogger.setup_master_logger()

    async def start(self, port: int):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=GRPC_OPTIONS)

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{port}")
        self.server.add_insecure_port(f"unix://{MASTER_SOCKET_PATH}")
        self.logger.info(f"Starting Master gRPC server on [::]:{port}")
        await self.server.start()

    async def stop(self):
        if self.server:
            try:
                await self.server.stop(grace=5)
                await self.server.wait_for_termination()
            except (Exception, asyncio.CancelledError) as e:
                self.logger.info("master handler error", str(e))

    async def Forward(
        self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """处理从最后一个节点返回的结果"""
        self.logger.info("master handler request")
        request_id = "-".join(x for x in list(request.uuid_list))

        try:
            self.pending_requests.complete_forward_request(request_id, request.hidden_states)
        except Exception as e:
            self.logger.debug("error")
        except BaseException as e:
            self.logger.debug("base error")
        finally:
            return schemas_pb2.ForwardResponse(msg="Forward Completed", status=200)

    async def ImageForward(
        self, request: schemas_pb2.ImageForwardRequest, context: grpc.ServicerContext
    ) -> schemas_pb2.ForwardResponse:
        """处理从最后一个节点返回的结果"""
        request_id = "-".join(x for x in list(request.uuid))

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
