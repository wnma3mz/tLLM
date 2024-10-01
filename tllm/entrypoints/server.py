import argparse
import json
import os
import time
from typing import *

from fastapi import FastAPI
import torch
import uvicorn

from tllm.engine import MyLlamaForCausalLM
from tllm.generate.decode_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.rpc.manager import RPCManager
from tllm.utils import setup_seed

app = FastAPI()


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # input_id_list = app.tok.preprocess_old(messages=request.messages).input_ids
    input_id_list = app.tok.preprocess(messages=request.messages).input_ids

    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    s1 = time.time()
    output = app.model.generate(
        input_ids, max_new_tokens=request.max_tokens, do_sample=request.do_sample, sampler=DecodeUtils("greedy")
    )
    print("output", output)
    return ChatCompletionResponse(
        token=output.output_ids,
        cost_time=time.time() - s1,
        finish_reason=output.finish_reason,
        usage={"prompt_tokens": len(input_id_list), "completion_tokens": len(output.output_ids)},
        text="",
    )


@app.post("/health")
async def health():
    return {"status": "ok"}


@app.post("/status")
async def status(request: Dict[str, float]):
    cost_time = request.get("cost_time", 0)
    pp_rank = request.get("pp_rank", 0)
    return {"cost_time": cost_time, "pp_rank": pp_rank}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--need_start_client", action="store_true")
    return parser.parse_args()


def start_client(config_path: str, model_path: str) -> None:
    # 启动 client
    with open(config_path, "r") as f:
        config_list = json.load(f)

    os.system("rm -rf grpc_*.log")
    for pp_config in config_list:
        port = pp_config["url"].rsplit(":", 1)[-1]
        start_layer_idx, end_layer_idx = pp_config["layer_idx"]
        # TODO 远程启动
        if pp_config["tp_size"] > 1:
            cmd = f"torchrun --nproc_per_node={pp_config['tp_size']} --master_port={pp_config['master_port']} tllm/rpc/client.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"
        else:
            # 几乎等效于 torchrun --nproc_per_node=1
            cmd = f"python3 tllm/rpc/client.py --start_layer_idx={start_layer_idx} --end_layer_idx={end_layer_idx} --model_path {model_path} --port {port} > grpc_{port}.log 2>&1 &"  #
        # 异步启动
        print(f"begin start client {pp_config['pp_rank']}")
        os.popen(cmd)
        # 监听是否启动成功
        while True:
            if os.path.exists(f"grpc_{port}.log"):
                with open(f"grpc_{port}.log", "r") as f:
                    if "Starting gRPC server on port" in f.read():
                        break
            time.sleep(1)
        print(f"start client {pp_config['pp_rank']} success")


def parse_url_list(config_path: str) -> List[str]:
    with open(config_path, "r") as f:
        config_list = json.load(f)
    return [pp_config["url"] for pp_config in config_list]


if __name__ == "__main__":
    setup_seed(42)
    args = parse_args()

    s1 = time.time()
    if args.need_start_client:
        start_client(args.config_path, args.model_path)
    url_list = parse_url_list(args.config_path)
    server = RPCManager(url_list)
    app.model = MyLlamaForCausalLM.from_pretrained(args.model_path, args.weight_path, server)
    app.tok = TokenizerUtils(args.model_path)
    print(f"init cost time {time.time() - s1}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
