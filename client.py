import argparse
import os
import time
from typing import *

from fastapi import FastAPI

from models.llama.decoder import Decoder
from models.llama.mlp import MLP
from tllm_schemas import ForwardData, LayerConfig, MLPConfig, MLPForwardData

app = FastAPI()

app.model = Decoder()
app.init_model_flag = False
app.mlp_dict = {}


@app.post("/init_model")
def init_model(data: LayerConfig):
    if app.init_model_flag:
        return {"msg": "Model already initialized", "status": 200}

    if not os.path.isfile(data.state_dict_path):
        return {"msg": "Model not found", "status": 404}

    s1 = time.time()
    print("Init Model Config")
    print(data)
    app.model.post_init(data)
    app.init_model_flag = True
    print(f"Model initialized cost time: {time.time() - s1:.2f} s")
    return {"msg": "Model initialized", "status": 200}


@app.post("/forward")
def forward(data: ForwardData):
    s1 = time.time()
    input_data = app.model._prepare_forward_data(data)
    output = app.model.forward(**input_data)
    return_output = app.model._prepare_output_data(data, output)
    print(f"Forward pass cost time: {time.time() - s1:.2f} s")
    return {"msg": "Forward pass completed", "status": 200, "output": return_output}


@app.post("/init_mlp")
def init_mlp(data: MLPConfig):
    data.__post_init__()
    if data.name in app.mlp_dict:
        return {"msg": "MLP already initialized", "status": 200}

    s1 = time.time()
    print(f"Init MLP {data.proj_name} Config: {data.input_size}x{data.output_size}")
    try:
        app.mlp_dict[data.name] = MLP(data)
    except Exception as e:
        print("MLP initialization failed ", e)
        return {"msg": "MLP initialization failed", "status": 500}
    print(f"MLP {data.name} initialized cost time: {time.time() - s1:.2f} s")
    return {"msg": "MLP initialized", "status": 200}


@app.post("/forward_mlp")
def forward_mlp(data: MLPForwardData):
    s1 = time.time()
    data.__post_init__()
    layer = app.mlp_dict[data.name]
    output = layer.forward(layer._prepare_forward_data(data))
    return_output = layer._prepare_output_data(output)
    print(f"Forward MLP {data.name} cost time: {time.time() - s1:.2f} s")
    return {"msg": "Forward MLP completed", "status": 200, "output": return_output}


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
