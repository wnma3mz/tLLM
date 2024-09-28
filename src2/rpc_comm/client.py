import argparse
from concurrent import futures
import json
import logging
import os
import time
from typing import *

from google.protobuf import json_format, struct_pb2
import grpc

from commons.communicator import Communicator
from llama import MyLlamaModel
from rpc_comm import schemas_pb2, schemas_pb2_grpc
from rpc_comm.convert import list_to_protobuf, protobuf_to_list
from schemas import ForwardData, LayerConfig
from utils import get_ip_address, tensor_to_list

logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist
from transformers import AutoConfig, LlamaForCausalLM

# PYTHONPATH="./src2:./src":$PYTHONPATH torchrun --nproc_per_node=2 src2/rpc_comm/client.py


class RPCServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, config, model, rank):
        self.config = config
        self.model = model
        self.rank = rank

        self.init_model_flag = False
        self.mlp_dict = {}
        self.int_key = [
            "bos_token_id",
            "eos_token_id",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pretraining_tp",
            "vocab_size",
        ]
        self.ip_addr = get_ip_address()
        self.prefix_log_str = f"IP: [{self.ip_addr}]"
        hidden_states_shape = torch.empty(3, dtype=torch.int64)
        uuid_shape = torch.empty(1, dtype=torch.int64)
        if self.config.comm.is_rank0():
            pass
        else:
            while True:
                dist.recv(hidden_states_shape, src=0)
                hidden_states = torch.empty(tuple(hidden_states_shape.tolist()))
                dist.recv(hidden_states, src=0)

                dist.recv(uuid_shape, src=0)
                uuid_tensor = torch.empty(tuple(uuid_shape.tolist()), dtype=torch.int8)
                dist.recv(uuid_tensor, src=0)
                uuid = "".join(chr(c) for c in uuid_tensor.numpy())

                input_data = self.model._prepare_forward_data_v2(hidden_states, uuid)
                output = self.model.forward(**input_data)

                self.model.cache_manager.set(uuid, output.past_key_values)
                self.model.cache_manager.check_alive()

    def Forward(self, request, context):
        s1 = time.time()
        hidden_states = protobuf_to_list(request.hidden_states)

        data = ForwardData(uuid=request.uuid, hidden_states=hidden_states)

        input_data = self.model._prepare_forward_data(data)
        shape = torch.tensor(input_data["hidden_states"].shape, dtype=torch.int64)
        uuid_tensor = torch.tensor([ord(c) for c in data.uuid], dtype=torch.int8)
        uuid_shape = torch.tensor(uuid_tensor.shape, dtype=torch.int64)
        for rank in range(1, self.config.comm.world_size):
            dist.send(shape, dst=rank)
            dist.send(input_data["hidden_states"], dst=rank)

            dist.send(uuid_shape, dst=rank)
            dist.send(uuid_tensor, dst=rank)

        output = self.model.forward(**input_data)

        return_output = tensor_to_list(self.model._prepare_output_data(request, output))
        return_output = list_to_protobuf(return_output)
        cost_time = time.time() - s1
        logging.info(f"{self.prefix_log_str} Forward pass cost time: {cost_time:.2f} s")
        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed",
            status=200,
            output=return_output,
            cost_time=cost_time,
        )

    def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)

    def InitModelFlag(self, request, context):
        return schemas_pb2.InitModelFlagResponse(msg=self.init_model_flag, status=200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    return parser.parse_args()


def start_grpc_server(config, model, port, rank):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    rpc_servicer = RPCServicer(config, model, rank)
    if config.comm.is_rank0():
        schemas_pb2_grpc.add_RPCServiceServicer_to_server(rpc_servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        print(f"Starting gRPC server on port {port}")
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    args = parse_args()
    comm = Communicator(is_torchrun=True)

    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.decoder_start_layer_idx = config.num_hidden_layers // 2
    config.decoder_end_layer_idx = config.num_hidden_layers
    config.comm = comm

    # s1 = time.time()
    # state_dict = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu").state_dict()
    # model = MyLlamaModel(config)
    # model.load_state_dict(state_dict)
    # print(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
    # model.eval()
    model = None

    start_grpc_server(config, model, args.port, comm.rank)
