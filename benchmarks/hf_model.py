import time
from typing import *

import torch
from transformers import LlamaForCausalLM

from tllm.generate.token_utils import TokenizerUtils

if __name__ == "__main__":
    model_path = "/Users/lujianghu/Documents/Llama-3.2-1B-Instruct"
    tok = TokenizerUtils(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu", torch_dtype=torch.bfloat16
    )

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tok.preprocess(messages=messages).input_ids
    input_ids = torch.tensor(input_id_list).unsqueeze(0)

    model.eval()
    print("input_ids: ", input_ids)

    # warmup
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print("generate token: ", output[0])

    # 模拟 TTFT 时间
    cost_time_list = []
    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        cost_time_list.append(time.time() - s1)
    cost_time_list = sorted(cost_time_list)[1:-1]
    ttft = sum(cost_time_list) / len(cost_time_list)
    print("ttft: ", ttft)

    # 模拟生成时间
    cost_time_list = []
    max_new_tokens = 20
    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        cost_time_list.append(time.time() - s1 - ttft)
    print("cost_time_list: ", cost_time_list)
    cost_time_list = sorted(cost_time_list)[1:-1]
    print("token/s: ", max_new_tokens / (sum(cost_time_list) / len(cost_time_list)))
