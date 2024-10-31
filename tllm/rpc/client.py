import argparse
from concurrent import futures
import os
import time
from typing import *
import uuid

import grpc
import torch
from transformers import AutoConfig

from tllm.commons.communicator import Communicator, SingleNodeCommunicator
from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.models.protocol import SeqInput
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.model_client import ModelClient
from tllm.utils import setup_logger


class RPCServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self, comm: Communicator, model, rank: int, pp_rank: int, ip_addr: str = "localhost"):
        self.comm = comm
        self.model = model
        self.rank = rank
        self.pp_rank = pp_rank
        self.init_model_flag = False
        self.ip_addr = ip_addr
        self.prefix_log_str = f"IP: [{self.ip_addr}]"
        uuid_shape_list = [None, None]
        if self.rank == 0:
            pass
        else:
            while self.comm.world_size > 1:
                self.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.model.dtype)
                self.comm.broadcast(hidden_states)

                _ = self.model.forward(hidden_states, seq_input=seq_input)

    def InitModel(self, request: schemas_pb2.ModelConfig, context: grpc.ServicerContext):
        self.init_model_flag = True
        self.master_url = request.master_url
        self.next_pp_rank = request.next_pp_rank
        return schemas_pb2.StatusResponse(msg="Init model completed", status=200)

    def Forward(self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext):
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: str
            seq_len: int
        """
        s1 = time.time()
        hidden_states = deserialize_tensor(request.hidden_states)

        seq_input = SeqInput(uuid_str_list=list(request.uuid), seq_len_list=list(request.seq_len))
        self.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
        self.comm.broadcast(hidden_states)

        output_hidden_states = self.model(hidden_states, seq_input)

        output = serialize_tensor(output_hidden_states)

        cost_time = time.time() - s1

        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed",
            status=200,
            output=output,
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
    parser.add_argument("--master_url", type=str, default="ws://localhost:8000")
    parser.add_argument("--start_layer_idx", type=int, default=0, help="start layer idx")
    parser.add_argument("--end_layer_idx", type=int, default=11, help="end layer idx")
    parser.add_argument("--pp_rank", type=int, default=0, help="pp rank")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--ip_addr", type=str, default="localhost", help="提供给 server 连接的 ip")
    return parser.parse_args()


def start_grpc_server(comm: Communicator, model, port: int, rank: int, pp_rank: int, ip_addr: str = "localhost"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    rpc_servicer = RPCServicer(comm, model, rank, pp_rank, ip_addr)
    if comm.is_rank0():
        schemas_pb2_grpc.add_RPCServiceServicer_to_server(rpc_servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        logger.info(f"Starting gRPC server on port {port}")
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger("client_" + __name__)
    comm = Communicator(is_torchrun=True) if "WORLD_SIZE" in os.environ else SingleNodeCommunicator()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config.comm = comm

    model_client = ModelClient(
        logger=logger,
        server_url=args.master_url,
        start_idx=args.start_layer_idx,
        end_idx=args.end_layer_idx,
        client_id=f"client-{str(uuid.uuid4())[:8]}-pp{args.start_layer_idx}-{args.end_layer_idx}",
        ip_addr=args.ip_addr,
        port=args.port,
    )
    model_client.start()
    model = model_client.load_model(config, args.model_path, torch.bfloat16)

    start_grpc_server(comm, model, args.port, comm.rank, args.pp_rank, args.ip_addr)
