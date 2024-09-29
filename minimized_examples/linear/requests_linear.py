import time
from typing import *

import grpc
import requests
import torch

from tllm.commons.convert import list_to_protobuf, protobuf_to_list
from tllm.rpc import schemas_pb2, schemas_pb2_grpc


def http_func():
    x = torch.randn(1, 4096).tolist()
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    json_data = {"x": x}
    s1 = time.time()
    s = requests.post("http://localhost:8003/forward", headers=headers, json=json_data)
    print("Cost time:", time.time() - s1)
    print(s.json()["cost_time"])


class RPCManager:
    def __init__(self, url_list: List[str]):
        self.stub_list = []
        url = "localhost:8004"
        channel = grpc.insecure_channel(url)
        self.stub_list.append(schemas_pb2_grpc.RPCServiceStub(channel))

    def forward(self, stub, data):
        x = torch.randn(1, 4096).tolist()
        s1 = time.time()
        request = schemas_pb2.ForwardData(uuid="1234t", hidden_states=list_to_protobuf(x))
        out = stub.Forward(request)
        print("Cost time:", time.time() - s1)
        print("Output:", out.cost_time)


if __name__ == "__main__":
    rpc_server = RPCManager(["localhost:8004"])
    rpc_server.forward(rpc_server.stub_list[0], {})
