from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen35DecoderLayer, TextModelArgs as Qwen35TextModelArgs

from tllm.commons.cache import AttentionData, cat_func
from tllm.models.backend_mlx.helper import MLXCacheManager


def empty_func(h, cache):
    return h


class Qwen35Decoder(nn.Module):
    def __init__(self, args: Qwen35TextModelArgs, start_layer_idx: int, end_layer_idx: int):
        super().__init__()
        self.start_layer_idx = start_layer_idx
        self.end_layer_idx = end_layer_idx
        self.layers = []
        for _ in range(start_layer_idx):
            self.layers.append(empty_func)
        for layer_idx in range(start_layer_idx, end_layer_idx):
            self.layers.append(Qwen35DecoderLayer(args, layer_idx))

        self.local_num_layers = end_layer_idx - start_layer_idx
        self.request_runtime_cache: Dict[str, list] = {}

    def _make_request_cache(self):
        cache = []
        for layer_idx in range(self.start_layer_idx, self.end_layer_idx):
            layer = self.layers[layer_idx]
            cache.append(ArraysCache(size=2) if layer.is_linear else KVCache())
        return cache

    def _get_request_runtime_cache(self, uuid: str):
        if uuid not in self.request_runtime_cache:
            self.request_runtime_cache[uuid] = self._make_request_cache()
        return self.request_runtime_cache[uuid]

    def _set_request_kv_len(self, request_cache, uuid: str):
        # Keep CacheManager sequence bookkeeping aligned with qwen3.5 runtime cache.
        local_cache = request_cache.get_decoder_cache(uuid)
        next_len = local_cache[0].kv_len + local_cache.q_len
        for layer_idx in range(self.local_num_layers):
            local_cache[layer_idx].set_kv_len(next_len)

    def __call__(self, h: mx.array, cache: AttentionData, cache_manager: MLXCacheManager) -> mx.array:
        request_cache = cache.request_cache
        active_uuid_set = set(cache.uuid_list)
        stale_uuid_list = [uuid for uuid in list(self.request_runtime_cache.keys()) if uuid not in active_uuid_set]
        for uuid in stale_uuid_list:
            if not cache_manager.contains(uuid):
                self.request_runtime_cache.pop(uuid, None)

        output_list = []
        start = 0
        for uuid in cache.uuid_list:
            q_len = request_cache.get_q_len(uuid)
            end = start + q_len
            hidden_states = h[start:end][None, ...]
            start = end

            runtime_cache = self._get_request_runtime_cache(uuid)
            fa_mask = None
            ssm_mask = None

            for local_idx, layer_idx in enumerate(range(self.start_layer_idx, self.end_layer_idx)):
                layer = self.layers[layer_idx]
                layer_cache = runtime_cache[local_idx]
                if layer.is_linear:
                    if ssm_mask is None:
                        ssm_mask = create_ssm_mask(hidden_states, layer_cache)
                    layer_mask = ssm_mask
                else:
                    if fa_mask is None:
                        fa_mask = create_attention_mask(hidden_states, layer_cache)
                    layer_mask = fa_mask
                hidden_states = layer(hidden_states, mask=layer_mask, cache=layer_cache)

            self._set_request_kv_len(request_cache, uuid)
            output_list.append(hidden_states.squeeze(0))
        return cat_func(output_list)
