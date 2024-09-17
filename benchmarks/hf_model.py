from transformers import LlamaForCausalLM, AutoTokenizer
from typing import *
import torch
import time


def load_model_and_tokenizer(model_path: str) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
    model = LlamaForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
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
    # print(tok.decode(1788))
    # assert False
    model.eval()

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    input_id_list = tokenize_message(tok, messages)
    input_ids = torch.tensor(input_id_list).unsqueeze(0)
    print("input_ids: ", input_ids)
    # output = model.generate(input_ids, max_new_tokens=50, tokenizer=tok, eos_token_id=[0, tok.eos_token_id])
    # print(tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
    print("generate token: ", output[0])

    for _ in range(0):
        s1 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        print(f"Time taken: {time.time() - s1}")
        print(tok.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True))

    # 2.6-3.0s
