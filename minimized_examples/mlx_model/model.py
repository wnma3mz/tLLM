import glob
import json
import os
from typing import Dict, Union

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.llama import ModelArgs
from transformers import AutoTokenizer

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


def load_config(model_path: str) -> ModelArgs:
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    model_args = ModelArgs.from_dict(config)
    return model_args


def generate_text(model_path: str):
    messages1 = [{"role": "user", "content": "Hello, how are you?"}]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    text = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    model, tokenizer = load(model_path)
    response = generate(model, tokenizer, prompt=text, verbose=True, max_tokens=10)
    print("response", response)


def build_model_input(model_path: str):
    messages1 = [{"role": "user", "content": "Hello, how are you?"}]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    text = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    return mx.array([tokenizer.encode(text)])


def base_generate(base_model, input_ids):
    base_out = base_model(input_ids)
    print("base_out", base_out.shape, base_out)


def load_my_model(model_args, weights):
    model_args.decoder_start_layer_idx = 0
    model_args.decoder_end_layer_idx = model_args.num_hidden_layers
    model = MyMLXLlamaModel(model_args)
    filter_w = {k: v for k, v in weights.items() if "model.layers." in k}
    model.load_weights(list(filter_w.items()))
    mx.eval(model.parameters())
    model.eval()
    return model


def forward_head(base_model, out):
    out = base_model.model.norm(out)
    logits = base_model.model.embed_tokens.as_linear(out)
    logits = logits[:, -1, :]
    return mx.argmax(logits, axis=-1).tolist()


if __name__ == "__main__":
    setup_seed(42)
    model_path = "/Users/jianghulu/Documents/Llama-3.2-1B-Instruct-bf16"
    # generate_text(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    weights = load_weight(model_path)
    model_args = load_config(model_path)

    base_model, tokenizer = load(model_path)
    input_ids = build_model_input(model_path)
    print("input_ids", input_ids.shape)

    embedding = base_model.model.embed_tokens(input_ids)

    model = load_my_model(model_args, weights)

    seq_input = SeqInput(["123"], [embedding.shape[1]])
    out = model(embedding, seq_input)
    idx = forward_head(base_model, out)
    print("idx", idx)
    print("token", tokenizer.decode(idx))

    # 生成第二个 token
    for _ in range(10):
        h2 = base_model.model.embed_tokens([idx])
        seq_input = SeqInput(["123"], [1])
        out = model(h2, seq_input)

        idx = forward_head(base_model, out)
        print("token", tokenizer.decode(idx))
