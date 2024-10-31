import asyncio
import os
import time
import uuid

import torch

from tllm.engine import AsyncEngine, MyLlamaForCausalLM, SequenceRequestData
from tllm.generate.sampler_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.rpc.manager import RPCManager


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def init_engine():
    url_list = ["localhost:25001"]
    BASE_PATH = "/Users/lujianghu/Documents/"
    model_path = os.path.join(BASE_PATH, "Llama-3.2-1B-Instruct")
    weight_path = os.path.join(model_path, "master_weight.pt")

    tok = TokenizerUtils(model_path)

    server = RPCManager(url_list)
    model = MyLlamaForCausalLM.from_pretrained(model_path, weight_path, server)
    engine = AsyncEngine(model)
    return engine, tok


async def fetch_result(result_generator: "AsyncIterator"):
    async for res in result_generator:
        print("=" * 20)
        final_res = res

    print(final_res)


async def main():
    engine, tok = init_engine()
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tok.preprocess(messages=messages).input_ids

    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print("input_ids: ", input_ids)

    s1 = time.time()
    request_id = f"chat-{random_uuid()}"
    sequence_data = SequenceRequestData(request_id=request_id, input_ids=input_ids, sampler=DecodeUtils("greedy"))
    result_generator = engine.generate_stream(sequence_data)
    fetch_result(result_generator)


if __name__ == "__main__":
    asyncio.run(main())
