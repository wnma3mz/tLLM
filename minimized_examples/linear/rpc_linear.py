from concurrent import futures
import time
from typing import *

from commons.convert import list_to_protobuf, protobuf_to_list
import grpc
from pydantic import BaseModel
from rpc_comm import schemas_pb2, schemas_pb2_grpc
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config: dict):
        super(MLP, self).__init__()
        self.proj = nn.Linear(config["input_size"], config["output_size"])

    def forward(self, x):
        return self.proj(x)


class MLPData(BaseModel):
    x: List[List[float]]


class RPCServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self):
        hidden_size = 4096
        self.layer = MLP({"input_size": hidden_size, "output_size": hidden_size})

    def Forward(self, request, context):
        s1 = time.time()
        hidden_states = protobuf_to_list(request.hidden_states)
        output = self.layer(torch.tensor(hidden_states))
        return_output = list_to_protobuf(output.tolist())
        cost_time = time.time() - s1
        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed",
            status=200,
            output=return_output,
            cost_time=cost_time,
        )


if __name__ == "__main__":
    host, port = "0.0.0.0", 8004
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    schemas_pb2_grpc.add_RPCServiceServicer_to_server(RPCServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
