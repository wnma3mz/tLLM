import glob
import os
import time
from typing import Dict, Union

import mlx.core as mx
from mlx_lm import generate, load
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from tllm.models.mlx_llama import MyMLXLlamaModel
from tllm.models.protocol import SeqInput


def setup_seed(seed: int):
    mx.random.seed(seed)


def load_weight(model_path: str) -> Dict[str, mx.array]:
    weight_files = glob.glob(os.path.join(model_path, "model*.safetensors"))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    return weights


def generate_text(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model, tokenizer = load(model_path)
    response = generate(model, tokenizer, prompt=text, verbose=False, max_tokens=new_token + 1)
    print(f"generate text: {response}")


def build_model_input(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return mx.array([tokenizer.encode(text)])


def base_generate(base_model, input_ids):
    base_out = base_model(input_ids)
    print("base_out", base_out.shape, base_out)


def load_my_model(config: AutoConfig, weights):
    config.decoder_start_layer_idx = 0
    config.decoder_end_layer_idx = config.num_hidden_layers
    model = MyMLXLlamaModel(config)
    filter_w = {k: v for k, v in weights.items() if "model.layers." in k}
    model.load_weights(list(filter_w.items()))
    mx.eval(model.parameters())
    model.eval()
    return model


def forward_head(base_model, out: mx.array):
    out = base_model.model.norm(out.astype(mx.bfloat16))
    logits = base_model.model.embed_tokens.as_linear(out)
    logits = logits[:, -1, :]
    return mx.argmax(logits, axis=-1).tolist()


if __name__ == "__main__":
    setup_seed(42)
    model_path = "/Users/jianghulu/Documents/Llama-3.2-1B-Instruct-bf16"
    new_token = 10
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    s1 = time.time()
    generate_text(model_path)
    print("mlx cost total time", time.time() - s1)
    print("=" * 20)
    # assert False
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    weights = load_weight(model_path)
    config = AutoConfig.from_pretrained(model_path)

    base_model, _ = load(model_path)
    input_ids = build_model_input(model_path)

    embedding = base_model.model.embed_tokens(input_ids)

    model = load_my_model(config, weights)

    _ = tokenizer.decode([0])  # for warmup

    seq_input = SeqInput(["123"], [embedding.shape[1]])
    s1 = time.perf_counter()
    out = model(embedding, seq_input)
    s3 = time.perf_counter()
    idx = forward_head(base_model, out)
    print("tllm head time", time.perf_counter() - s3)
    s2 = time.perf_counter()

    text = tokenizer.decode(idx)
    print("tllm ttft time", s2 - s1)
    # 生成第二个 token
    for _ in range(new_token):
        h2 = base_model.model.embed_tokens([idx])
        seq_input = SeqInput(["123"], [1])
        out = model(h2, seq_input)

        idx = forward_head(base_model, out)
        text += tokenizer.decode(idx)
    print(f"generate text: {text}")
    print("tllm tpot time", time.perf_counter() - s2)
    # print("tllm tpot time", new_token / (time.perf_counter() - s2))
    print(f"tllm cost total time", time.perf_counter() - s1)

    # I'm just a large language model, I don't have personal feelings or
    # I'm happy to help with any questions or topics you'd like to discuss
    # 我想问你关于天气的天气预测，通常天
