import asyncio
from dataclasses import dataclass
import os

from tllm.engine import SequenceRequestData
from tllm.entrypoints.protocol import random_uuid
from tllm.generate.sampler_utils import SamplerUtils, SamplingParams
from tllm.rpc.client import run
from tllm.utils import init_engine


@dataclass
class Args:
    model_path: str = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    config_path: str = "./examples/config.json"
    weight_path: str = None

    def __post_init__(self):
        self.weight_path = os.path.join(self.model_path, "master_weight.pt")


@dataclass
class ClientArgs:
    host: str = "0.0.0.0"
    port: int = 50051
    master_url: str = "ws://localhost:8000"
    start_layer_idx: int = 0
    end_layer_idx: int = 16
    pp_rank: int = 0
    model_path: str = "/Users/jianghulu/Documents/Llama-3.2-1B-Instruct-bf16"
    ip_addr: str = "localhost"


async def fetch_result(result_generator: "AsyncIterator"):
    async for res in result_generator:
        print("=" * 20)
        final_res = res

    print(final_res)


async def main():
    client_args = ClientArgs()
    run(client_args, True)
    args = Args()
    engine, tok = init_engine(args)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_ids = tok.preprocess(messages=messages).input_ids

    print("input_ids: ", input_ids)

    request_id = f"chat-{random_uuid()}"
    sequence_data = SequenceRequestData(
        request_id=request_id,
        input_ids=input_ids,
        sampler=SamplerUtils("greedy", tok),
        sampling_params=SamplingParams(max_tokens=16),
    )
    result_generator = await engine.generate(sequence_data)

    # result_generator = engine.generate_stream(sequence_data)
    # await fetch_result(result_generator)


if __name__ == "__main__":
    asyncio.run(main())
