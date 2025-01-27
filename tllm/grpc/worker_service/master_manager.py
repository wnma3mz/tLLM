# coding: utf-8
from typing import List, Tuple

import grpc

from tllm.grpc.proto import schemas_pb2, schemas_pb2_grpc


class MasterRPCManager:
    # 向 Master 发送 gRPC 请求
    # 每个节点需要发送处理完之后的「结果」和「状态」
    # 「结果」发往下一个 PP，如果已经是最后一个 PP，则发往 Master 节点
    # 「状态」发往 Master 节点
    def __init__(self, grpc_options: List[Tuple[str, int]]):
        self.grpc_options = grpc_options
        self.master_stub = None

    def update_url(self, master_url: str, forward_url: str, pp_idx: int):
        master_channel = grpc.aio.insecure_channel(master_url, options=self.grpc_options)
        forward_channel = grpc.aio.insecure_channel(forward_url, options=self.grpc_options)
        self.master_stub = schemas_pb2_grpc.RPCServiceStub(master_channel)
        self.forward_stub = schemas_pb2_grpc.RPCServiceStub(forward_channel)
        self.pp_idx = pp_idx

    async def rpc_func(
        self, request: schemas_pb2.ForwardRequest, hidden_states: schemas_pb2.BFloat16Tensor, cost_time: float
    ):
        forward_request = {
            "uuid_list": request.uuid_list,
            "hidden_states": hidden_states,
            "input_ids_list": request.input_ids_list,
        }
        status_request = {
            "uuid": request.uuid_list,
            "pp_idx": self.pp_idx,
            "cost_time": cost_time,
        }
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
        }
        status_request = {"uuid": request.uuid, "pp_idx": self.pp_idx, "cost_time": cost_time}
        await self.master_stub.Status(schemas_pb2.StatusRequest(**status_request))
        await self.forward_stub.ImageForward(schemas_pb2.ImageForwardRequest(**forward_request))
