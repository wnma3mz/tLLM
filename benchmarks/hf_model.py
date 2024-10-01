import time
from typing import *

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def load_model_and_tokenizer(model_path: str) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
    # https://gist.github.com/wnma3mz/64db3e69616b819de346635b7bfa1d36
    # model.load_custom_weights(model.state_dict())
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tok


formatted_prompt = "### Human: {}### Assistant:"


def tokenize_message(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> List[int]:
    inputs = formatted_prompt.format(messages[0]["content"])
    # inputs = "Hello, how are you?"
    # inputs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(inputs, add_special_tokens=True)
    while input_ids[0] == input_ids[1] == tok.bos_token_id:
        # input_ids = input_ids[1:]
        input_ids.pop(0)
    return input_ids


if __name__ == "__main__":
    # model_path = "/Users/jianghulu/Documents/TinyLlama-1.1B-Chat-v0.1"
    model_path = "/Users/lujianghu/Documents/TinyLlama-1.1B-Chat-v1.0"
    model, tok = load_model_and_tokenizer(model_path)
    model.eval()

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print("input_ids: ", input_ids)
    # output = model.generate(input_ids, max_new_tokens=50, tokenizer=tok, eos_token_id=[0, tok.eos_token_id])
    # print(tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print("generate token: ", output[0])

    time_list = []
    max_new_tokens = 20
    for _ in range(10):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        time_list.append(time.time() - s1)
        # print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))
    print("token/s: ", max_new_tokens / (sum(time_list) / len(time_list)))