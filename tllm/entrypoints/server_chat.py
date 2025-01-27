import asyncio
import os
import time
from typing import AsyncIterator

from fastapi import HTTPException, Request

from tllm.engine import AsyncEngine
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
from tllm.generate import MessageProcessor
from tllm.schemas import RequestOutput, SequenceRequestData


def create_error_response(message: str) -> ChatCompletionResponse:
    raise HTTPException(status_code=499, detail=message)


class OpenAIServing:
    def __init__(self, engine: AsyncEngine, args):
        self.engine = engine
        self.message_processor = MessageProcessor(self.engine.tok)
        self.model_name = os.path.basename(args.model_path)
        self.response_role = "assistant"

    @property
    def max_model_len(self):
        return 8192

    async def show_available_models(self):
        model_cards = [
            ModelCard(id=self.model_name, max_model_len=self.max_model_len, root="tllm", permission=[ModelPermission()])
        ]
        return ModelList(data=model_cards)

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        request_id = f"chat-{random_uuid()}"
        messages, mm_input_dict = await self.message_processor.parse_message(request.messages)
        input_ids = self.message_processor.preprocess(messages)

        if request.temperature == 0.0:
            method = "greedy"
        else:
            method = "sampling"

        sequence_data = SequenceRequestData(
            request_id=request_id,
            sampling_params=request.to_sampling_params(self.engine.tok.tokenizer),
            input_ids=input_ids,
            multi_modal_inputs=mm_input_dict,
        )
        result_generator = self.engine.generate_stream(sequence_data)

        if request.stream:
            return self.chat_completion_stream_generator(request, raw_request, request_id, result_generator)
        else:
            return await self.chat_completion_full_generator(request, raw_request, request_id, result_generator)

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        request_id: str,
        result_generator: AsyncIterator,
    ) -> AsyncIterator[str]:
        created_time = int(time.time())
        n = 1
        previous_texts = [""] * n
        try:
            async for res in result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(request_id)
                    create_error_response("Client disconnected")
                res: RequestOutput
                output = res.outputs[0]
                i = output.index

                delta_text = output.text
                previous_texts[i] = output.text

                # 根据 finish_reason 判断是否结束，分别处理
                if output.finish_reason is not None:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, delta=DeltaMessage(content=""), logprobs=None, finish_reason=output.finish_reason
                    )
                else:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        logprobs=None,
                        finish_reason=output.finish_reason,
                    )
                chunk = ChatCompletionStreamResponse(
                    id=request_id, model=self.model_name, created=created_time, choices=[choice_data]
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            create_error_response("Client disconnected")

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
        request_id: str,
        result_generator: AsyncIterator,
    ) -> ChatCompletionResponse:
        final_res = None
        created_time = int(time.time())
        try:
            async for res in result_generator:
                if raw_request is not None and await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(request_id)
                    create_error_response("Client disconnected")
                await asyncio.sleep(0.01)
                final_res: RequestOutput = res
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            create_error_response("Client disconnected")

        output = final_res.outputs[0]
        message = ChatMessage(role=self.response_role, content=output.text)

        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            logprobs=None,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
        )

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        total_tokens = num_prompt_tokens + num_generated_tokens

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=total_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=self.model_name,
            choices=[choice_data],
            usage=usage,
        )

        return response
