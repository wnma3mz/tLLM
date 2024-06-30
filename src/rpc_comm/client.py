import argparse
import json
import logging
import os
import time
from concurrent import futures
from typing import *

import grpc
from google.protobuf import json_format, struct_pb2

from models.llama.decoder import Decoder

# from models.llama.mlp import MLP
from rpc_comm import schemas_pb2, schemas_pb2_grpc
from rpc_comm.convert import list_to_protobuf, protobuf_to_list
from schemas import ForwardData, LayerConfig, MLPConfig, MLPForwardData
from utils import get_ip_address

logging.basicConfig(level=logging.INFO)


class RPCServicer(schemas_pb2_grpc.RPCServiceServicer):
    def __init__(self):
        self.model = Decoder()
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

    def InitModel(self, request, context):
        if self.init_model_flag:
            return schemas_pb2.StatusResponse(msg="Model already initialized", status=200)

        json_string = json_format.MessageToJson(request.config)
        json_config = json.loads(json_string)
        for key in self.int_key:
            if key in json_config:
                json_config[key] = int(json_config[key])
        data = LayerConfig(
            config=json_config,
            layer_idx_start=request.layer_idx_start,
            layer_idx_end=request.layer_idx_end,
            tp_url_list=request.tp_url_list,
            tp_size=request.tp_size,
            layer_state_dict_dir=request.layer_state_dict_dir,
        )
        if not os.path.isdir(data.layer_state_dict_dir):
            return schemas_pb2.StatusResponse(msg="Model not found", status=404)

        s1 = time.time()
        logging.info(f"{self.prefix_log_str} Init Model Config")
        logging.info(f"{self.prefix_log_str} {data}")
        self.model.post_init(data)
        self.init_model_flag = True
        cost_time = time.time() - s1
        logging.info(f"{self.prefix_log_str} Model initialized cost time: {cost_time:.2f} s")
        return schemas_pb2.StatusResponse(msg="Model initialized", status=200)

    def Forward(self, request, context):
        s1 = time.time()
        hidden_states = protobuf_to_list(request.hidden_states)
        data = ForwardData(uuid=request.uuid, hidden_states=hidden_states)
        input_data = self.model._prepare_forward_data(data)
        output = self.model.forward(**input_data)
        return_output = self.model._prepare_output_data(request, output)
        return_output = list_to_protobuf(return_output)
        cost_time = time.time() - s1
        logging.info(f"{self.prefix_log_str} Forward pass cost time: {cost_time:.2f} s")
        return schemas_pb2.ForwardResponse(
            msg="Forward pass completed", status=200, output=return_output, cost_time=cost_time
        )

    def InitMLP(self, request, context):
        data = MLPConfig(
            proj_name=request.proj_name,
            input_size=request.input_size,
            output_size=request.output_size,
            state_dict_path=request.state_dict_path,
        )
        data.__post_init__()
        if data.name in self.mlp_dict:
            return schemas_pb2.StatusResponse(msg="MLP already initialized", status=200)

        s1 = time.time()
        logging.info(f"{self.prefix_log_str} Init MLP {data.proj_name} Config: {data.input_size}x{data.output_size}")
        try:
            pass
            # self.mlp_dict[data.name] = MLP(data)
        except Exception as e:
            logging.info(f"{self.prefix_log_str} MLP initialization failed ", e)
            return schemas_pb2.StatusResponse(msg="MLP initialization failed", status=500)
        logging.info(f"{self.prefix_log_str} MLP {data.name} initialized cost time: {time.time() - s1:.2f} s")
        return schemas_pb2.StatusResponse(msg="MLP initialized", status=200)

    def ForwardMLP(self, request, context):
        s1 = time.time()
        hidden_states = protobuf_to_list(request.hidden_states)
        data = MLPForwardData(
            proj_name=request.proj_name, hidden_states=hidden_states, tp_idx=request.tp_idx, layer_idx=request.layer_idx
        )
        data.__post_init__()
        layer = self.mlp_dict[data.name]
        output = layer.forward(layer._prepare_forward_data(request))
        return_output = layer._prepare_output_data(output)
        return_output = list_to_protobuf(return_output)
        cost_time = time.time() - s1
        logging.info(f"{self.prefix_log_str} Forward MLP {data.name} cost time: {cost_time} s")
        return schemas_pb2.ForwardResponse(
            msg="Forward MLP completed", status=200, output=return_output, cost_time=cost_time
        )

    def Health(self, request, context):
        return schemas_pb2.HealthResponse(msg="Healthy", status=200)

    def MLPKeys(self, request, context):
        return schemas_pb2.MLPKeysResponse(msg=str(list(self.mlp_dict.keys())), status=200)

    def InitModelFlag(self, request, context):
        return schemas_pb2.InitModelFlagResponse(msg=self.init_model_flag, status=200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    schemas_pb2_grpc.add_RPCServiceServicer_to_server(RPCServicer(), server)
    # server.add_insecure_port(f"{args.host}:{args.port}")
    server.add_insecure_port(f"[::]:{args.port}")
    logging.info(f"Starting gRPC server on {args.host}:{args.port}")
    server.start()
    server.wait_for_termination()
