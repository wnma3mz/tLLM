import copy
import json
import os

import mlx.core as mx
from mlx_lm.models.llama import ModelArgs

from tllm.commons.cache import CacheManager
from tllm.models.mlx.layers import MyAttention
from tllm.models.mlx_llama import build_forward_cache, build_mlx_mask
from tllm.models.protocol import SeqInput
from tllm.models.utils import build_mask


def test_build_mask():
    mask = build_mask([(2, 3), (1, 4)])
    mlx_mask = build_mlx_mask([(2, 3), (1, 4)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)

    prefill_seq_len = 3
    mask = build_mask([(prefill_seq_len, prefill_seq_len)])
    mlx_mask = build_mlx_mask([(prefill_seq_len, prefill_seq_len)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)

    mask = build_mask([(1, 1 + prefill_seq_len)])
    mlx_mask = build_mlx_mask([(1, 1 + prefill_seq_len)])
    print("mask", mask)
    print("mlx_mask", mlx_mask)


def test_mlp():
    pass


def setup_seed(seed: int):
    mx.random.seed(seed)


def build_forward(seq_input, hidden_states, attn):
    cache_manager = CacheManager()
    attention_cache_list = build_forward_cache(seq_input, cache_manager, 1)
    cache = attention_cache_list[0]
    mask = cache.attn_mask
    mask = mask if mask is None else mask.astype(hidden_states.dtype)
    output = attn(hidden_states, mask=mask, cache=cache)
    return output, cache


def build_cache_forward(seq_input, hidden_states, attn, cache_manager):
    attention_cache_list = build_forward_cache(seq_input, cache_manager, 1)
    cache = attention_cache_list[0]
    mask = cache.attn_mask
    # print("mask", mask)
    if len(seq_input.uuid_list) > 1:
        # print("mask", mask)
        # print("mask[0]", mask[0][:5])
        # print("mask[1]", mask[1][:5])
        new_mask = mask
        # new_mask = mx.zeros_like(mask)
        # print("new_mask", new_mask)
        output = attn(hidden_states, mask=new_mask, cache=cache)
    else:
        mask = mask if mask is None else mask.astype(hidden_states.dtype)
        output = attn(hidden_states, mask=mask, cache=cache)
    return output


if __name__ == "__main__":
    # init
    setup_seed(42)
    model_path = "/Users/jianghulu/Documents/Llama-3.2-1B-Instruct-bf16"
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    args = ModelArgs.from_dict(config)
    attn = MyAttention(args)
    mx.eval(attn.parameters())
    attn.eval()
    bsz = 1

    # prefilling request 1
    seq_len = 2
    h1 = mx.random.uniform(shape=(bsz, seq_len, args.hidden_size))
    seq_input = SeqInput(uuid_list=["0"], seq_len_list=[seq_len])
    o1, cache1 = build_forward(seq_input, h1, attn)

    # prefilling request 2
    seq_len = 3
    h2 = mx.random.uniform(shape=(bsz, seq_len, args.hidden_size))
    seq_input = SeqInput(uuid_list=["1"], seq_len_list=[seq_len])
    o2, cache2 = build_forward(seq_input, h2, attn)
    base_o12 = mx.concat([o1, o2], axis=1)

    # prefilling multi request
    seq_input = SeqInput(uuid_list=["2", "3"], seq_len_list=[2, 3])
    cat_o, cache12 = build_forward(seq_input, mx.concat([h1, h2], axis=1), attn)

    print("output", mx.allclose(base_o12, cat_o, 1e-4), abs(base_o12 - cat_o).sum())
    print("=" * 20)

    # decoding request 1
    # cache_manager = CacheManager()

    # h11 = mx.random.uniform(shape=(bsz, 1, args.hidden_size))
    # seq_input = SeqInput(uuid_list=["0"], seq_len_list=[1])
    # cache_manager.set("0", [cache1.past_key_value.get_cache("0")])
    # o11 = build_cache_forward(seq_input, h11, attn, cache_manager)
    # print("o11", o11)

    # # decoding multi request
    # h1122 = mx.concat([h11, h2], axis=1)
    # seq_input = SeqInput(uuid_list=["0", "4"], seq_len_list=[1, 3])
    # o1122 = build_cache_forward(seq_input, h1122, attn, cache_manager)
    # print("o1122", o1122)
    # assert False

    # decoding request 1
    cache_manager = CacheManager()

    h11 = mx.random.uniform(shape=(bsz, 1, args.hidden_size))
    seq_input = SeqInput(uuid_list=["0"], seq_len_list=[1])
    cache_manager.set("0", [cache1.past_key_value.get_cache("0")])
    # copy 数据，防止 cache 被修改
    cache_manager.set("2", [copy.deepcopy(cache1.past_key_value.get_cache("0"))])
    a = cache_manager.cache_dict["0"]["past_key_values"][0].keys
    o11 = build_cache_forward(seq_input, h11, attn, cache_manager)
    print("o11", o11)

    # decoding request 2
    h21 = mx.random.uniform(shape=(bsz, 1, args.hidden_size))
    seq_input = SeqInput(uuid_list=["1"], seq_len_list=[1])
    cache_manager.set("1", [cache2.past_key_value.get_cache("1")])
    cache_manager.set("3", [copy.deepcopy(cache2.past_key_value.get_cache("1"))])
    o21 = build_cache_forward(seq_input, h21, attn, cache_manager)
    print("o21", o21)

    # decoding multi request
    h1122 = mx.concat([h11, h21], axis=1)

    seq_input = SeqInput(uuid_list=["2", "3"], seq_len_list=[1, 1])
    o1122 = build_cache_forward(seq_input, h1122, attn, cache_manager)
    print("o1122", o1122)

    bast_cat_o = mx.concat([o11, o21], axis=1)
    print("output", mx.allclose(o1122, bast_cat_o, 1e-4), abs(o1122 - bast_cat_o).sum())
