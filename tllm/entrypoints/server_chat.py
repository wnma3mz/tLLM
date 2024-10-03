import json
import logging
import os
import time
from typing import *

from fastapi import Request
import torch

from tllm.engine import MyLlamaForCausalLM
from tllm.generate.decode_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.protocol import ChatCompletionRequest, ChatCompletionResponse
from tllm.rpc.manager import RPCManager


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


class OpenAIServing:

    def __init__(self, args):
        url_list = parse_url_list(args.config_path)
        server = RPCManager(url_list)
        self.model = MyLlamaForCausalLM.from_pretrained(args.model_path, args.weight_path, server)
        self.tok = TokenizerUtils(args.model_path)

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):

        input_id_list = self.tok.preprocess(messages=request.messages).input_ids

        input_ids = torch.tensor(input_id_list).unsqueeze(0)
        result_generator = self.model.generate(
            input_ids, max_new_tokens=request.max_tokens, do_sample=request.do_sample, sampler=DecodeUtils("greedy")
        )
        raw_request.prompt_tokens = len(input_id_list)

        if request.stream:
            return self.chat_completion_stream_generator(request, raw_request, result_generator)
        else:
            return await self.chat_completion_full_generator(request, raw_request, result_generator)

    async def chat_completion_stream_generator(
        self, request: ChatCompletionRequest, raw_request: Request, result_generator: AsyncIterator
    ) -> AsyncIterator[str]:
        s1 = time.time()
        async for res in result_generator:
            chunk = ChatCompletionResponse(
                token=res.output_ids,
                cost_time=time.time() - s1,
                ttft=res.ttft,
                finish_reason=res.finish_reason,
                usage={"prompt_tokens": raw_request.prompt_tokens, "completion_tokens": 1},
                text="",
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self, request: ChatCompletionRequest, raw_request: Request, result_generator: AsyncIterator
    ) -> ChatCompletionResponse:
        output_token = []
        final_res = None
        s1 = time.time()
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                return self.create_error_response("Client disconnected")
            output_token.append(res.output_ids)
            final_res = res

        response = ChatCompletionResponse(
            token=output_token,
            cost_time=time.time() - s1,
            ttft=final_res.ttft,
            finish_reason=final_res.finish_reason,
            usage={"prompt_tokens": raw_request.prompt_tokens, "completion_tokens": len(output_token)},
            text="",
        )
        return response
