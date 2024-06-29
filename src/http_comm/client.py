import argparse
import logging
import os
import time
from typing import *

from fastapi import FastAPI

from models.llama.decoder import Decoder
from models.llama.mlp import MLP
from schemas import ForwardData, LayerConfig, MLPConfig, MLPForwardData
from utils import get_ip_address

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.model = Decoder()
app.init_model_flag = False
app.mlp_dict = {}
app.prefix_log_str = f"IP: [{get_ip_address()}]"


@app.post("/init_model")
def init_model(data: LayerConfig):
    if app.init_model_flag:
        return {"msg": "Model already initialized", "status": 200}

    if not os.path.isdir(data.layer_state_dict_dir):
        return {"msg": "Model not found", "status": 404}

    s1 = time.time()
    logging.info(f"{app.prefix_log_str} Init Model Config")
    logging.info(f"{app.prefix_log_str} {data}")
    app.model.post_init(data)
    app.init_model_flag = True
    logging.info(f"{app.prefix_log_str} Model initialized cost time: {time.time() - s1:.2f} s")
    return {"msg": "Model initialized", "status": 200}


@app.post("/forward")
def forward(data: ForwardData):
    s1 = time.time()
    input_data = app.model._prepare_forward_data(data)
    output = app.model.forward(**input_data)
    return_output = app.model._prepare_output_data(data, output)
    cost_time = time.time() - s1
    logging.info(f"{app.prefix_log_str} Forward pass cost time: {cost_time:.2f} s")
    return {"msg": "Forward pass completed", "status": 200, "output": return_output, "cost_time": cost_time}


@app.post("/init_mlp")
def init_mlp(data: MLPConfig):
    data.__post_init__()
    if data.name in app.mlp_dict:
        return {"msg": "MLP already initialized", "status": 200}

    s1 = time.time()
    logging.info(f"{app.prefix_log_str} Init MLP {data.proj_name} Config: {data.input_size}x{data.output_size}")
    try:
        app.mlp_dict[data.name] = MLP(data)
    except Exception as e:
        logging.info(f"{app.prefix_log_str} MLP initialization failed ", e)
        return {"msg": "MLP initialization failed", "status": 500}
    logging.info(f"{app.prefix_log_str} MLP {data.name} initialized cost time: {time.time() - s1:.2f} s")
    return {"msg": "MLP initialized", "status": 200}


@app.post("/forward_mlp")
def forward_mlp(data: MLPForwardData):
    s1 = time.time()
    data.__post_init__()
    layer = app.mlp_dict[data.name]
    output = layer.forward(layer._prepare_forward_data(data))
    return_output = layer._prepare_output_data(output)
    cost_time = time.time() - s1
    logging.info(f"{app.prefix_log_str} Forward MLP {data.name} cost time: {cost_time:.2f} s")
    return {"msg": "Forward MLP completed", "status": 200, "output": return_output, "cost_time": cost_time}


@app.get("/health")
def health():
    return {"msg": "Healthy", "status": 200}


@app.get("/mlp_keys")
def mlp_keys():
    return {"msg": list(app.mlp_dict.keys()), "status": 200}


@app.get("/init_model_flag")
def init_model_flag():
    return {"msg": app.init_model_flag, "status": 200}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
