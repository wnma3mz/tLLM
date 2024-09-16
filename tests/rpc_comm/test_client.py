import json
from typing import *

import grpc
from google.protobuf import json_format, struct_pb2
from transformers import AutoConfig

from rpc_comm import schemas_pb2, schemas_pb2_grpc

# gRPC 服务端地址和端口
SERVER_ADDRESS = "localhost:50051"

from rpc_comm.convert import list_to_protobuf, protobuf_to_list


def test_init_model():
    # 创建 gRPC 通道
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    # 创建一个 gRPC 客户端存根
    stub = schemas_pb2_grpc.RPCServiceStub(channel)

    config = AutoConfig.from_pretrained(
        "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    )
    config_struct_obj = struct_pb2.Struct()
    json_format.Parse(json.dumps(config.to_dict()), config_struct_obj)
    # 构造 InitModel 请求
    layer_config = schemas_pb2.LayerConfig(
        config=config_struct_obj,
        layer_idx_start=0,
        layer_idx_end=16,
        tp_url_list=[""],
        tp_size=1,
        layer_state_dict_dir="./weights",
    )

    # 调用 InitModel RPC 方法
    response = stub.InitModel(layer_config)
    print("InitModel Response:", response.msg, response.status)


def test_forward():
    # 创建 gRPC 通道
    channel = grpc.insecure_channel(SERVER_ADDRESS)

    # 创建一个 gRPC 客户端存根
    stub = schemas_pb2_grpc.RPCServiceStub(channel)

    import torch

    hidden_states = torch.rand((1, 2, 2048)).tolist()
    # 构造 ForwardData 请求
    forward_data = schemas_pb2.ForwardData(
        uuid="123456", hidden_states=list_to_protobuf(hidden_states)
    )

    # 调用 Forward RPC 方法
    response = stub.Forward(forward_data)
    print("Forward Response:", response.msg, response.status)


# 测试代码的入口
if __name__ == "__main__":
    test_init_model()
    test_forward()
