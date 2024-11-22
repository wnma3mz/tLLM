import glob
import json
import os
from typing import Dict, Union

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.base import create_attention_mask, create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.llama import ModelArgs
from transformers import AutoTokenizer

from tllm.commons.cache import AttentionData, RequestsCache
from tllm.models.mlx_llama import Decoder


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


def my_create_attention_mask(h: mx.array, cache: Union[KVCache, RotatingKVCache]) -> mx.array:
    T = h.shape[1]
    if T > 1:
        window_size = None
        offset = 0
        c = cache
        if hasattr(c, "max_size"):
            offset = min(c.max_size - 1, c.offset)
            window_size = c.max_size
        else:
            offset = c.offset
        mask = create_causal_mask(T, offset, window_size=window_size)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


def generate_text(model_path: str):
    messages1 = [{"role": "user", "content": "Hello, how are you?"}]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    text = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    model, tokenizer = load(model_path)
    response = generate(model, tokenizer, prompt=text, verbose=True)
    print("response", response)


if __name__ == "__main__":
    setup_seed(42)
    model_path = "/Users/jianghulu/Documents/Llama-3.2-1B-Instruct-bf16"

    weights = load_weight(model_path)
    model_args = load_config(model_path)

    model = Decoder(model_args, 0, 1)
    filter_w = {k.split("model.")[-1]: v for k, v in weights.items() if "model.layers.0" in k}
    model.load_weights(list(filter_w.items()))

    mx.eval(model.parameters())
    model.eval()

    bsz, seq_len, hidden_size = 1, 4, model_args.hidden_size

    num_layers = 1
    request_cache = RequestsCache()
    request_cache.add("123", 4, None)
    uuid_list = ["123"]
    cache = [
        AttentionData(
            request_cache=request_cache,
            uuid_list=uuid_list,
            attn_mask=None,  # TODO mlx create_attention_mask
        )
    ]
    one_layer_cache = cache[0].past_key_value.cache_dict[uuid_list[0]]["cache"]
    print(one_layer_cache)

    # bsz x seq_len x hidden_size
    h = mx.random.uniform(0, 1, (bsz, seq_len, hidden_size))
    mask = my_create_attention_mask(h, one_layer_cache)
    out = model(h, mask, cache)

    base_model, tokenizer = load(model_path)
    base_out = base_model.layers[0](h, mask, one_layer_cache)

    assert mx.allclose(out, base_out)
