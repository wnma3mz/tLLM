from concurrent.futures import ThreadPoolExecutor
import json
from typing import *

from google.protobuf import json_format, struct_pb2
import grpc

from tllm.rpc import schemas_pb2, schemas_pb2_grpc


class RPCManager:
    def __init__(self, url_list: List[Optional[str]], pp_size: int = 1):
        self.stub_list = []
        for url in url_list:
            if url is not None:
                channel = grpc.insecure_channel(url)
                self.stub_list.append(schemas_pb2_grpc.RPCServiceStub(channel))
            else:
                self.stub_list.append(None)
        self.executor = ThreadPoolExecutor()
        self.pp_size = pp_size
        self.func_dict = {
            "forward": self.forward,
            "init_model": self.init_model,
            "health": self.health,
            "init_model_flag": self.init_model_flag,
        }

    @property
    def size(self):
        return len(self.stub_list)

    def update_url(self, pp_idx: int, url: str) -> bool:
        if pp_idx >= len(self.stub_list):
            return False
        self.stub_list[pp_idx] = schemas_pb2_grpc.RPCServiceStub(grpc.insecure_channel(url))
        return True

    def remove_url(self, pp_idx: int) -> bool:
        if pp_idx >= len(self.stub_list):
            return False
        self.stub_list[pp_idx] = None
        return True

    def is_full_connected(self) -> bool:
        return all([stub is not None for stub in self.stub_list])

    def init_model(self, stub, data):
        config_struct_obj = struct_pb2.Struct()
        json_format.Parse(json.dumps(data["config"]), config_struct_obj)
        request = schemas_pb2.ModelConfig(
            model_name=data["model_name"],
            pp_rank=data["pp_rank"],
            layer_idx_start=data["layer_idx_start"],
            layer_idx_end=data["layer_idx_end"],
            master_url=data["master_url"],
            next_pp_rank=data["next_pp_rank"],
        )
        return stub.InitModel(request)

    def forward(self, stub, data):
        request = schemas_pb2.ForwardRequest(**data)
        return stub.Forward(request)

    def health(self, stub):
        request = schemas_pb2.Empty()
        return stub.Health(request)

    def init_model_flag(self, stub):
        request = schemas_pb2.Empty()
        return stub.InitModelFlag(request)

    def __len__(self):
        return len(self.stub_list)

    # 异步 post
    def post(self, path, data_list: List[Dict[str, Any]]):
        if path[0] == "/":
            path = path[1:]
        response_list = []
        for stub, data in zip(self.stub_list, data_list):
            response = self.func_dict[path](stub, data)
            response_list.append(response)
        return response_list

    # 单个 post
    def post_sync(self, url_idx: int, path, data):
        if path[0] == "/":
            path = path[1:]
        stub = self.stub_list[url_idx]
        return self.func_dict[path](stub, data)

    def is_success(self, response_list) -> bool:
        if isinstance(response_list, list):
            return all([response.status == 200 for response in response_list])
        return response_list.status == 200


if __name__ == "__main__":
    # for test
    server = RPCManager(["localhost:50051", "localhost:50052"])
    server.post_sync(0, "/forward", {"uuid": "123", "hidden_states": [[1.0, 2.0, 3.0]]})
