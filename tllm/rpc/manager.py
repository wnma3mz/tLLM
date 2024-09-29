from concurrent.futures import ThreadPoolExecutor
import json
from typing import *

from commons.convert import list_to_protobuf, protobuf_to_list
from google.protobuf import json_format, struct_pb2
import grpc
from rpc import schemas_pb2, schemas_pb2_grpc


class RPCManager:
    def __init__(self, url_list: List[str]):
        self.stub_list = []
        self.url_list = url_list
        for url in url_list:
            channel = grpc.insecure_channel(url)
            self.stub_list.append(schemas_pb2_grpc.RPCServiceStub(channel))
        self.executor = ThreadPoolExecutor()
        self.func_dict = {
            "init_model": self.init_model,
            "forward": self.forward,
            "health": self.health,
            "init_model_flag": self.init_model_flag,
        }

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
        request = schemas_pb2.ForwardRequest(uuid=data["uuid"], hidden_states=list_to_protobuf(data["hidden_states"]))
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

    # 指定 api path，对所有 url 发送 post 请求
    def post_thread(self, path, data_list: List[Dict[str, Any]]) -> List:
        if path[0] == "/":
            path = path[1:]
        response_list = []
        futures = []

        for stub, data in zip(self.stub_list, data_list):
            future = self.executor.submit(self.func_dict[path], stub, data)
            futures.append(future)

        for future in futures:
            response_list.append(future.result())
        return response_list

    # 指定 url_idx，多线程请求
    def post_thread_url(self, url_idx, path, data_list: List[Dict[str, Any]]) -> List:
        if path[0] == "/":
            path = path[1:]
        response_list = []
        futures = []
        stub = self.stub_list[url_idx]
        for data in data_list:
            future = self.executor.submit(self.func_dict[path], stub, data)
            futures.append(future)

        for future in futures:
            response_list.append(future.result())
        return response_list

    # 指定 api path, url_idx 以及每个 url 的请求 data_list 多线程请求
    def post_thread_url_dict(self, path: str, stub_idx_data_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List]:
        if path[0] == "/":
            path = path[1:]

        # 返回结果按照 dict 顺序
        response_dict = {}
        futures = []

        for stub_idx, data_list in stub_idx_data_dict.items():
            for data in data_list:
                future = self.executor.submit(self.func_dict[path], self.stub_list[stub_idx], json=data)
                futures.append(future)

        for stub_idx, data_list in stub_idx_data_dict.items():
            response_dict[stub_idx] = []
            for _ in data_list:
                response_dict[stub_idx].append(futures.pop(0).result())
        return response_dict

    # 单个 post
    def post_sync(self, url_idx: int, path, data):
        if path[0] == "/":
            path = path[1:]
        stub = self.stub_list[url_idx]
        return self.func_dict[path](stub, data)

    def fetch_list_output(self, response_list) -> List:
        if isinstance(response_list, list):
            return [protobuf_to_list(response.output) for response in response_list]
        return protobuf_to_list(response_list.output)

    def is_success(self, response_list) -> bool:
        if isinstance(response_list, list):
            return all([response.status == 200 for response in response_list])
        return response_list.status == 200

    def fetch_list_cost_time(self, response_list) -> List:
        if isinstance(response_list, list):
            return [response.cost_time for response in response_list]
        return response_list.cost_time


if __name__ == "__main__":
    # for test
    server = RPCManager(["localhost:50051", "localhost:50052"])
    server.post_sync(0, "/forward", {"uuid": "123", "hidden_states": [[1.0, 2.0, 3.0]]})
