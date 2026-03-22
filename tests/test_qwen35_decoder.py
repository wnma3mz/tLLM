import importlib.util
from pathlib import Path
import sys
import types

import mlx.core as mx

from tllm.commons.cache import AttentionData

if "mlx_lm.models.base" not in sys.modules:
    fake_base = types.ModuleType("mlx_lm.models.base")
    fake_base.create_attention_mask = lambda *args, **kwargs: None
    fake_base.create_ssm_mask = lambda *args, **kwargs: None
    sys.modules["mlx_lm.models.base"] = fake_base

if "mlx_lm.models.cache" not in sys.modules:
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    class FakeArraysCache:
        def __init__(self, size, left_padding=None):
            self.cache = [None] * size
            self.left_padding = left_padding
            self.lengths = None

        def __getitem__(self, idx):
            return self.cache[idx]

        def __setitem__(self, idx, value):
            self.cache[idx] = value

        def empty(self):
            return self.cache[0] is None

        def prepare(self, lengths=None, **kwargs):
            self.lengths = lengths

        def extract(self, idx):
            out = FakeArraysCache(len(self.cache))
            out.cache = [None if c is None else c[idx : idx + 1] for c in self.cache]
            return out

        @classmethod
        def merge(cls, caches):
            merged = cls(len(caches[0].cache))
            for state_idx in range(len(merged.cache)):
                init = next((c[state_idx] for c in caches if c[state_idx] is not None), None)
                if init is None:
                    continue
                shape = list(init.shape)
                shape[0] = len(caches)
                merged[state_idx] = mx.zeros(shape, dtype=init.dtype)
                for batch_idx, cache in enumerate(caches):
                    if cache[state_idx] is not None:
                        merged[state_idx][batch_idx : batch_idx + 1] = cache[state_idx]
            return merged

    class FakeBatchKVCache:
        def __init__(self, left_padding):
            self.left_padding = left_padding

        @classmethod
        def merge(cls, caches):
            return cls(left_padding=[0] * len(caches))

        def prepare(self, **kwargs):
            return None

        def finalize(self):
            return None

        def extract(self, idx):
            return self

        def empty(self):
            return True

    class FakeKVCache:
        def empty(self):
            return True

    fake_cache.ArraysCache = FakeArraysCache
    fake_cache.BatchKVCache = FakeBatchKVCache
    fake_cache.KVCache = FakeKVCache
    sys.modules["mlx_lm.models.cache"] = fake_cache

if "mlx_lm.models.qwen3_5" not in sys.modules:
    fake_qwen35 = types.ModuleType("mlx_lm.models.qwen3_5")
    fake_qwen35.DecoderLayer = object
    fake_qwen35.TextModelArgs = object
    sys.modules["mlx_lm.models.qwen3_5"] = fake_qwen35

decoder_path = Path(__file__).resolve().parents[1] / "tllm" / "models" / "qwen3_5" / "decoder.py"
decoder_spec = importlib.util.spec_from_file_location("test_qwen35_decoder_module", decoder_path)
decoder_module = importlib.util.module_from_spec(decoder_spec)
assert decoder_spec.loader is not None
decoder_spec.loader.exec_module(decoder_module)
Qwen35Decoder = decoder_module.Qwen35Decoder


class FakeLogger:
    def debug(self, *args, **kwargs):
        return None


class FakeSequenceCache:
    def __init__(self):
        self.kv_len = 0

    def set_kv_len(self, kv_len: int):
        self.kv_len = kv_len


class FakeDecoderCache:
    def __init__(self, q_len: int, num_layers: int):
        self.q_len = q_len
        self.kv_cache_list = [FakeSequenceCache() for _ in range(num_layers)]

    def __getitem__(self, idx: int):
        return self.kv_cache_list[idx]


class FakeRequestCache:
    def __init__(self, q_len_map, num_layers: int):
        self.decoder_cache_map = {
            uuid: FakeDecoderCache(q_len=q_len, num_layers=num_layers) for uuid, q_len in q_len_map.items()
        }

    def get_q_len(self, uuid: str) -> int:
        return self.decoder_cache_map[uuid].q_len

    def get_decoder_cache(self, uuid: str):
        return self.decoder_cache_map[uuid]


class FakeCacheManager:
    def contains(self, uuid: str) -> bool:
        return False


class RecordingLayer:
    def __init__(self, is_linear: bool):
        self.is_linear = is_linear
        self.calls = []

    def __call__(self, x, mask=None, cache=None):
        self.calls.append(
            {
                "shape": tuple(x.shape),
                "cache_id": id(cache),
                "mask_is_none": mask is None,
            }
        )
        return x


def build_decoder(layer):
    decoder = Qwen35Decoder.__new__(Qwen35Decoder)
    decoder.start_layer_idx = 0
    decoder.end_layer_idx = 1
    decoder.layers = [layer]
    decoder.local_num_layers = 1
    decoder.request_runtime_cache = {}
    decoder.logger = FakeLogger()
    decoder._continuous_uuid_order = None
    decoder._continuous_layer_cache = None
    decoder._q1_batch_buffer = None
    return decoder


def make_attention_data(uuid_list, q_len_map):
    return AttentionData(
        uuid_list=uuid_list,
        request_cache=FakeRequestCache(q_len_map=q_len_map, num_layers=1),
        attn_mask=None,
    )


def test_linear_layer_mixed_q_len_falls_back_to_single_path():
    layer = RecordingLayer(is_linear=True)
    decoder = build_decoder(layer)
    h = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    cache = make_attention_data(["req-1", "req-2"], {"req-1": 2, "req-2": 1})

    output = decoder(h, cache, FakeCacheManager())

    assert tuple(output.shape) == (3, 4)
    assert [call["shape"] for call in layer.calls] == [(1, 2, 4), (1, 1, 4)]
    assert decoder._continuous_layer_cache is None


def test_linear_layer_mixed_q_len_batches_same_length_requests_together():
    layer = RecordingLayer(is_linear=True)
    decoder = build_decoder(layer)
    h = mx.arange(20, dtype=mx.float32).reshape(5, 4)
    cache = make_attention_data(["req-1", "req-2", "req-3"], {"req-1": 2, "req-2": 1, "req-3": 2})

    output = decoder(h, cache, FakeCacheManager())

    assert tuple(output.shape) == (5, 4)
    assert [call["shape"] for call in layer.calls] == [(2, 2, 4), (1, 1, 4)]
    assert decoder._continuous_layer_cache is None


def test_q_len_1_continuous_batch_reuses_merged_linear_cache():
    layer = RecordingLayer(is_linear=True)
    decoder = build_decoder(layer)

    first_h = mx.arange(8, dtype=mx.float32).reshape(2, 4)
    first_cache = make_attention_data(["req-1", "req-2"], {"req-1": 1, "req-2": 1})
    decoder(first_h, first_cache, FakeCacheManager())

    assert [call["shape"] for call in layer.calls] == [(2, 1, 4)]
    assert decoder._continuous_uuid_order == ("req-1", "req-2")
    first_cache_id = layer.calls[0]["cache_id"]

    second_h = mx.arange(8, 16, dtype=mx.float32).reshape(2, 4)
    second_cache = make_attention_data(["req-1", "req-2"], {"req-1": 1, "req-2": 1})
    decoder(second_h, second_cache, FakeCacheManager())

    assert [call["shape"] for call in layer.calls] == [(2, 1, 4), (2, 1, 4)]
    assert layer.calls[1]["cache_id"] == first_cache_id
