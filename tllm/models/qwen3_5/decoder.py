from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen35DecoderLayer, TextModelArgs as Qwen35TextModelArgs

from tllm.commons.cache import AttentionData, cat_func
from tllm.models.backend_mlx.helper import MLXCacheManager
from tllm.singleton_logger import SingletonLogger


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
        self.logger = SingletonLogger.setup_master_logger()

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

    @staticmethod
    def _cache_is_mergeable(layer_cache) -> bool:
        if layer_cache is None:
            return False
        empty_func = getattr(layer_cache, "empty", None)
        if callable(empty_func):
            return not empty_func()
        return True

    @staticmethod
    def _pad_batch_hidden_states(hidden_states_list: List[mx.array], q_len_list: List[int]) -> mx.array:
        max_q_len = max(q_len_list)
        padded_hidden_states = []
        for hidden_states, q_len in zip(hidden_states_list, q_len_list):
            pad_len = max_q_len - q_len
            if pad_len == 0:
                padded_hidden_states.append(hidden_states)
                continue
            pad = mx.zeros((1, pad_len, hidden_states.shape[-1]), dtype=hidden_states.dtype)
            padded_hidden_states.append(mx.concatenate([hidden_states, pad], axis=1))
        return mx.concatenate(padded_hidden_states, axis=0)

    @staticmethod
    def _all_caches_empty(layer_cache_group: List) -> bool:
        for layer_cache in layer_cache_group:
            empty_func = getattr(layer_cache, "empty", None)
            if not callable(empty_func) or not empty_func():
                return False
        return True

    @staticmethod
    def _should_merge_layer(q_len_list: List[int], layer_cache_group: List) -> bool:
        batch_size = len(q_len_list)
        if batch_size < 2:
            return False
        if not all(
            Qwen35Decoder._cache_is_mergeable(c) for c in layer_cache_group
        ) and not Qwen35Decoder._all_caches_empty(layer_cache_group):
            return False

        # TPOT-like path: always merge for q_len == 1.
        if max(q_len_list) == 1:
            return True

        # TTFT/mixed path: merge only when padding overhead is limited.
        total_tokens = sum(q_len_list)
        max_q_len = max(q_len_list)
        pad_ratio = (batch_size * max_q_len) / max(total_tokens, 1)
        return pad_ratio <= 1.35

    @staticmethod
    def _run_layer_single(
        layer,
        hidden_states_list: List[mx.array],
        runtime_cache_list: List[List],
        local_idx: int,
    ) -> None:
        for req_idx, hidden_states in enumerate(hidden_states_list):
            layer_cache = runtime_cache_list[req_idx][local_idx]
            layer_mask = (
                create_ssm_mask(hidden_states, layer_cache)
                if layer.is_linear
                else create_attention_mask(hidden_states, layer_cache)
            )
            hidden_states_list[req_idx] = layer(hidden_states, mask=layer_mask, cache=layer_cache)

    def __call__(self, h: mx.array, cache: AttentionData, cache_manager: MLXCacheManager) -> mx.array:
        request_cache = cache.request_cache
        active_uuid_set = set(cache.uuid_list)
        stale_uuid_list = [uuid for uuid in list(self.request_runtime_cache.keys()) if uuid not in active_uuid_set]
        for uuid in stale_uuid_list:
            if not cache_manager.contains(uuid):
                self.request_runtime_cache.pop(uuid, None)

        hidden_states_list: List[mx.array] = []
        runtime_cache_list = []
        q_len_list = []
        start = 0
        for uuid in cache.uuid_list:
            q_len = request_cache.get_q_len(uuid)
            end = start + q_len
            hidden_states = h[start:end][None, ...]
            start = end

            hidden_states_list.append(hidden_states)
            runtime_cache_list.append(self._get_request_runtime_cache(uuid))
            q_len_list.append(q_len)

        for local_idx, layer_idx in enumerate(range(self.start_layer_idx, self.end_layer_idx)):
            layer = self.layers[layer_idx]
            layer_cache_group = [runtime_cache_list[idx][local_idx] for idx in range(len(hidden_states_list))]

            if not self._should_merge_layer(q_len_list, layer_cache_group):
                self.logger.debug(
                    "[qwen3_5 decoder] single layer=%d is_linear=%s batch=%d q_lens=%s",
                    layer_idx,
                    layer.is_linear,
                    len(hidden_states_list),
                    q_len_list,
                )
                self._run_layer_single(layer, hidden_states_list, runtime_cache_list, local_idx)
                continue

            batch_hidden_states = self._pad_batch_hidden_states(hidden_states_list, q_len_list)
            right_padding = [batch_hidden_states.shape[1] - q_len for q_len in q_len_list]

            if layer.is_linear:
                if self._all_caches_empty(layer_cache_group):
                    batch_cache = ArraysCache(size=2)
                else:
                    batch_cache = ArraysCache.merge(layer_cache_group)
                batch_cache.prepare(lengths=q_len_list)
            else:
                if self._all_caches_empty(layer_cache_group):
                    batch_cache = BatchKVCache(left_padding=[0] * len(layer_cache_group))
                else:
                    batch_cache = BatchKVCache.merge(layer_cache_group)
                batch_cache.prepare(right_padding=right_padding)

            self.logger.debug(
                "[qwen3_5 decoder] merge layer=%d is_linear=%s batch=%d q_lens=%s",
                layer_idx,
                layer.is_linear,
                len(hidden_states_list),
                q_len_list,
            )
            batch_mask = (
                create_ssm_mask(batch_hidden_states, batch_cache)
                if layer.is_linear
                else create_attention_mask(batch_hidden_states, batch_cache)
            )
            batch_output = layer(batch_hidden_states, mask=batch_mask, cache=batch_cache)
            batch_cache.finalize()

            for req_idx, q_len in enumerate(q_len_list):
                hidden_states_list[req_idx] = batch_output[req_idx : req_idx + 1, :q_len, :]
                runtime_cache_list[req_idx][local_idx] = batch_cache.extract(req_idx)

        for uuid in cache.uuid_list:
            self._set_request_kv_len(request_cache, uuid)
        return cat_func([x.squeeze(0) for x in hidden_states_list])
