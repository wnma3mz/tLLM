import argparse
from concurrent import futures
import logging
import os
import time
from typing import *

import grpc
import torch
from transformers import AutoConfig

from tllm.commons.communicator import Communicator, SingleNodeCommunicator
from tllm.commons.convert import deserialize_tensor, serialize_tensor
from tllm.rpc import schemas_pb2, schemas_pb2_grpc
from tllm.rpc.model_client import ClientArgs, ModelClient
from tllm.schemas import SeqInput
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
        self.server = None
        if self.rank == 0:
            pass
        else:
            while self.comm.world_size > 1:
                self.comm.broadcast_object(uuid_shape_list)
                seq_input, hidden_states_shape = uuid_shape_list
                hidden_states = torch.empty(hidden_states_shape, dtype=self.model.dtype)
                self.comm.broadcast(hidden_states)

                _ = self.model.forward(hidden_states, seq_input=seq_input)

    def start(self):
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_metadata_size", 32 * 1024 * 1024),
                ("grpc.max_send_message_length", 128 * 1024 * 1024),
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ],
        )

        schemas_pb2_grpc.add_RPCServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port(f"[::]:{args.port}")
        self.logger.info(f"Starting gRPC server on port {args.port}")
        self.server.start()

    def stop(self):
        if self.server:
            try:
                self.server.stop(grace=5)
                self.server.wait_for_termination()
            except Exception as e:
                pass

    def Forward(self, request: schemas_pb2.ForwardRequest, context: grpc.ServicerContext):
        """
        @param request: ForwardRequest
            hidden_states: bytes
            uuid: str
            seq_len: int
        """
        s1 = time.perf_counter()
        hidden_states = deserialize_tensor(request.hidden_states)

        seq_input = SeqInput(uuid_list=list(request.uuid), seq_len_list=list(request.seq_len))
        self.comm.broadcast_object([seq_input, tuple(hidden_states.shape)])
        self.comm.broadcast(hidden_states)
        print(f"deserialize_tensor cost time: {time.perf_counter() - s1:.4f}")

        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        cost_time = time.perf_counter() - s1
        print(f"forward cost time: {cost_time:.4f}")

        s1 = time.perf_counter()
        output = serialize_tensor(output_hidden_states)
        print(f"serialize_tensor cost time: {time.perf_counter() - s1:.4f}")
        print("=" * 20)
        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed",
            status=200,
            output=output,
            cost_time=cost_time,
        )

    def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)


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


def run(args):
    comm = SingleNodeCommunicator()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        comm = Communicator(is_torchrun=True)
    logger = setup_logger("client_" + __name__, logging.DEBUG)

    client_args = ClientArgs(
        start_layer_idx=args.start_layer_idx,
        end_layer_idx=args.end_layer_idx,
        ip_addr=args.ip_addr,
        port=args.port,
        master_url=args.master_url,
    )
    model_client = ModelClient(logger=logger, args=client_args)
    model_client.start()
    model = model_client.load_model(comm, args.model_path)

    rpc_servicer = RPCServicer(comm, model, comm.rank, args.pp_rank, args.ip_addr)
    if comm.rank == 0:
        rpc_servicer.start()


if __name__ == "__main__":
    args = parse_args()
    run(args)
