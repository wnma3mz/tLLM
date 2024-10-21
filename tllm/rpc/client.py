import argparse
from concurrent import futures
import logging
import os
import time
from typing import *

import grpc

from tllm.commons.communicator import Communicator, SingleNodeCommunicator
from tllm.commons.convert import deserialize_bfloat16_tensor, serialize_bfloat16_tensor
from tllm.models.llama import MyLlamaModel
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.protocol import SeqInput

logging.basicConfig(level=logging.INFO)

import torch
from transformers import AutoConfig, LlamaForCausalLM


class RPCServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, config, model, rank: int, pp_rank: int):
        self.config = config
        self.model = model
        self.rank = rank
        self.pp_rank = pp_rank
        self.init_model_flag = False
        self.ip_addr = "localhost"
        self.prefix_log_str = f"IP: [{self.ip_addr}]"
        uuid_shape_list = [None, None]
        if self.rank == 0:
            pass
        else:
            while self.comm.world_size > 1:
                self.config.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.model.dtype)
                self.config.comm.broadcast(hidden_states)

                _ = self.model.forward(hidden_states, seq_input=seq_input)

    def InitModel(self, request: schemas_pb2.ModelConfig, context: grpc.ServicerContext):
        """
        初始化模型，并 load 权重，需要同步至其他 TP
        @param request: ModelConfig
            model_name: str
            pp_rank: int
            layer_idx_start: int
            layer_idx_end: int
            master_url: str
            next_pp_rank: int
        """
        self.init_model_flag = True
        self.master_url = request.master_url
        self.next_pp_rank = request.next_pp_rank
        return schemas_pb2.StatusResponse(msg="Init model completed", status=200)

    def Forward(self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext):
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: str
        """
        s1 = time.time()
        hidden_states = deserialize_bfloat16_tensor(request.hidden_states)

        seq_input = SeqInput(uuid_str_list=list(request.uuid), seq_len_list=list(request.seq_len))
        self.config.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
        self.config.comm.broadcast(hidden_states)

        output = self.model(hidden_states, seq_input)

        return_output = serialize_bfloat16_tensor(output)
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
    parser.add_argument("--start_layer_idx", type=int, default=0, help="start layer idx")
    parser.add_argument("--end_layer_idx", type=int, default=11, help="end layer idx")
    parser.add_argument("--pp_rank", type=int, default=0, help="pp rank")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    return parser.parse_args()


def start_grpc_server(config, model, port, rank, pp_rank):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    rpc_servicer = RPCServicer(config, model, rank, pp_rank)
    if config.comm.is_rank0():
        schemas_pb2_grpc.add_RPCServiceServicer_to_server(rpc_servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        logging.info(f"Starting gRPC server on port {port}")
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    args = parse_args()
    comm = Communicator(is_torchrun=True) if "WORLD_SIZE" in os.environ else SingleNodeCommunicator()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.decoder_start_layer_idx = args.start_layer_idx
    config.decoder_end_layer_idx = args.end_layer_idx
    config.comm = comm

    dtype = torch.bfloat16
    s1 = time.time()
    state_dict = LlamaForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, device_map="cpu", torch_dtype=dtype, low_cpu_mem_usage=True
    ).state_dict()
    model = MyLlamaModel(config).to(dtype)
    model.load_state_dict(state_dict)
    logging.info(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
    model.eval()
    del state_dict

    start_grpc_server(config, model, args.port, comm.rank, args.pp_rank)
