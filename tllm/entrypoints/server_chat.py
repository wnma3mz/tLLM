import json
import logging
import os
import time
from typing import *

from fastapi import Request

from tllm.engine import AsyncEngine, RequestOutput, SequenceRequestData
from tllm.entrypoints.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    random_uuid,
)
from tllm.generate.decode_utils import DecodeUtils


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

    def __init__(self, engine: AsyncEngine, args):
        self.engine = engine
        self.model_name = os.path.basename(args.model_path)

    async def show_available_models(self):
        model_cards = [ModelCard(id=self.model_name, max_model_len=8192, root="tllm", permission=[ModelPermission()])]
        return ModelList(data=model_cards)

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        request_id = f"chat-{random_uuid()}"
        input_ids = self.engine.preprocess(request.messages)

        sequence_data = SequenceRequestData(request_id=request_id, input_ids=input_ids, sampler=DecodeUtils("greedy"))
        result_generator = self.engine.generate_stream(sequence_data)

        if request.stream:
            return self.chat_completion_stream_generator(request, raw_request, request_id, result_generator)
        else:
            return await self.chat_completion_full_generator(request, raw_request, request_id, result_generator)

    async def chat_completion_stream_generator(
        self, request: ChatCompletionRequest, raw_request: Request, request_id: str, result_generator: AsyncIterator
    ) -> AsyncIterator[str]:
        created_time = int(time.time())
        n = 1
        previous_texts = [""] * n
        async for res in result_generator:
            res: RequestOutput
            output = res.outputs[0]
            i = output.index

            delta_text = output.text
            previous_texts[i] = output.text

            # 最后一次返回为空字符串，且 finish_reason 不能为 None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(content=delta_text), logprobs=None, finish_reason=output.finish_reason
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id, model=self.model_name, created=created_time, choices=[choice_data]
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self, request: ChatCompletionRequest, raw_request: Request, request_id: str, result_generator: AsyncIterator
    ) -> ChatCompletionResponse:
        final_res = None
        role = "assistant"
        created_time = int(time.time())
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                return self.create_error_response("Client disconnected")
            final_res = res

        output = final_res.outputs[0]
        message = ChatMessage(role=role, content=output.text)

        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            logprobs=None,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
        )

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=self.model_name,
            choices=[choice_data],
            usage=usage,
        )
        return response
