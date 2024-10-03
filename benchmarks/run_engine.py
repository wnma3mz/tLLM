import os
import time

import torch

from tllm.engine import MyLlamaForCausalLM
from tllm.entrypoints.protocol import ChatCompletionRequest
from tllm.generate.decode_utils import DecodeUtils
from tllm.generate.token_utils import TokenizerUtils
from tllm.rpc.manager import RPCManager

if __name__ == "__main__":
    BASE_PATH = "/Users/lujianghu/Documents/"
    model_path = os.path.join(BASE_PATH, "Llama-3.2-1B-Instruct")
    weight_path = os.path.join(model_path, "master_weight.pt")

    url_list = ["localhost:25001"]
    server = RPCManager(url_list)
    model = MyLlamaForCausalLM.from_pretrained(model_path, weight_path, server)
    tok = TokenizerUtils(model_path)

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello, how are you?"}], max_tokens=20, do_sample=False
    )
    input_id_list = tok.preprocess(messages=request.messages).input_ids

    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print("input_ids: ", input_ids)

    s1 = time.time()
    output = model.generate(
        input_ids, max_new_tokens=request.max_tokens, do_sample=request.do_sample, sampler=DecodeUtils("greedy")
    )
    print(output)
