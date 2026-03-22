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
        # Stable decode path cache (continuous batching): reuse merged layer caches
        self._continuous_uuid_order = None
        self._continuous_layer_cache = None
        # Reusable input buffer for common q_len=1 path
        self._q1_batch_buffer = None

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

    def _pad_batch_hidden_states(
        self, hidden_states_list: List[mx.array], q_len_list: List[int], max_q_len: int
    ) -> mx.array:
        if all(q_len == max_q_len for q_len in q_len_list):
            if max_q_len == 1:
                batch_size = len(hidden_states_list)
                hidden_size = hidden_states_list[0].shape[-1]
                shape = (batch_size, 1, hidden_size)
                if (
                    self._q1_batch_buffer is None
                    or self._q1_batch_buffer.shape != shape
                    or self._q1_batch_buffer.dtype != hidden_states_list[0].dtype
                ):
                    self._q1_batch_buffer = mx.zeros(shape, dtype=hidden_states_list[0].dtype)
                for idx, hidden_states in enumerate(hidden_states_list):
                    self._q1_batch_buffer[idx : idx + 1] = hidden_states
                return self._q1_batch_buffer
            return mx.concatenate(hidden_states_list, axis=0)
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
    def _can_merge_layer_batch(q_len_list: List[int], layer_cache_group: List, is_linear: bool) -> bool:
        if len(layer_cache_group) < 2:
            return False
        if is_linear and len(set(q_len_list)) != 1:
            # Right-padding is safe for attention caches, but linear-attention keeps
            # a rolling conv state. Padding zeros would leak into that state and make
            # batched TTFT diverge from the per-request path.
            return False
        if not all(
            Qwen35Decoder._cache_is_mergeable(c) for c in layer_cache_group
        ) and not Qwen35Decoder._all_caches_empty(layer_cache_group):
            return False
        return True

    @staticmethod
    def _group_request_indices_by_q_len(q_len_list: List[int]) -> List[List[int]]:
        # Linear-attention can only safely share state within equal-length groups.
        q_len_group_map: Dict[int, List[int]] = {}
        for req_idx, q_len in enumerate(q_len_list):
            q_len_group_map.setdefault(q_len, []).append(req_idx)
        return list(q_len_group_map.values())

    @staticmethod
    def _run_layer_per_request(
        layer,
        hidden_states_list: List[mx.array],
        layer_cache_group: List,
        request_indices: List[int],
        ssm_mask_cache: List[mx.array],
        fa_mask_cache: List[mx.array],
    ) -> None:
        for req_idx, layer_cache in zip(request_indices, layer_cache_group):
            hidden_states = hidden_states_list[req_idx]
            if layer.is_linear:
                if ssm_mask_cache[req_idx] is None:
                    ssm_mask_cache[req_idx] = create_ssm_mask(hidden_states, layer_cache)
                layer_mask = ssm_mask_cache[req_idx]
            else:
                if fa_mask_cache[req_idx] is None:
                    fa_mask_cache[req_idx] = create_attention_mask(hidden_states, layer_cache)
                layer_mask = fa_mask_cache[req_idx]
            hidden_states_list[req_idx] = layer(hidden_states, mask=layer_mask, cache=layer_cache)

    def _run_linear_layer_grouped_batch(
        self,
        layer,
        hidden_states_list: List[mx.array],
        runtime_cache_list: List[List],
        local_idx: int,
        q_len_list: List[int],
        ssm_mask_cache: List[mx.array],
    ) -> None:
        # Mixed-q_len linear layers are bucketed by q_len so merged execution never
        # advances a request with padded fake steps.
        for request_indices in self._group_request_indices_by_q_len(q_len_list):
            layer_cache_group = [runtime_cache_list[idx][local_idx] for idx in request_indices]
            group_q_len_list = [q_len_list[idx] for idx in request_indices]
            if not self._can_merge_layer_batch(group_q_len_list, layer_cache_group, is_linear=True):
                self.logger.debug(
                    "[qwen3_5 decoder] linear per-request layer=%d batch=%d q_lens=%s",
                    self.start_layer_idx + local_idx,
                    len(request_indices),
                    group_q_len_list,
                )
                self._run_layer_per_request(
                    layer,
                    hidden_states_list,
                    layer_cache_group=layer_cache_group,
                    request_indices=request_indices,
                    ssm_mask_cache=ssm_mask_cache,
                    fa_mask_cache=None,
                )
                continue

            batch_hidden_states = mx.concatenate([hidden_states_list[idx] for idx in request_indices], axis=0)
            if self._all_caches_empty(layer_cache_group):
                batch_cache = ArraysCache(size=2)
            else:
                batch_cache = ArraysCache.merge(layer_cache_group)
            batch_cache.prepare(lengths=group_q_len_list)
            batch_mask = create_ssm_mask(batch_hidden_states, batch_cache)
            batch_output = layer(batch_hidden_states, mask=batch_mask, cache=batch_cache)

            self.logger.debug(
                "[qwen3_5 decoder] linear grouped-merge layer=%d batch=%d q_lens=%s",
                self.start_layer_idx + local_idx,
                len(request_indices),
                group_q_len_list,
            )
            for batch_idx, req_idx in enumerate(request_indices):
                q_len = q_len_list[req_idx]
                hidden_states_list[req_idx] = batch_output[batch_idx : batch_idx + 1, :q_len, :]
                runtime_cache_list[req_idx][local_idx] = batch_cache.extract(batch_idx)

    def _reset_continuous_state(self):
        self._continuous_uuid_order = None
        self._continuous_layer_cache = None

    def _materialize_continuous_state(self, runtime_cache_list: List[List]):
        if self._continuous_layer_cache is None:
            return
        for local_idx, batch_cache in enumerate(self._continuous_layer_cache):
            for req_idx in range(len(runtime_cache_list)):
                runtime_cache_list[req_idx][local_idx] = batch_cache.extract(req_idx)

    def _can_use_continuous_batch_state(self, uuid_list: List[str], q_len_list: List[int]) -> bool:
        if self._continuous_layer_cache is None or self._continuous_uuid_order is None:
            return False
        if any(cache is None for cache in self._continuous_layer_cache):
            return False
        if tuple(uuid_list) != self._continuous_uuid_order:
            return False
        # Reusing a merged decode cache is only safe when every request advances by
        # one real token; otherwise linear-attention state would drift.
        return max(q_len_list) == 1

    def _run_merged_layer_batch(
        self,
        layer,
        layer_idx: int,
        local_idx: int,
        hidden_states_list: List[mx.array],
        runtime_cache_list: List[List],
        q_len_list: List[int],
        max_q_len: int,
        right_padding: List[int],
        use_continuous_batch_state: bool,
        merge_ssm_mask,
        merge_fa_mask,
    ):
        # This path covers attention mixed batches and the equal-length decode fast
        # path. Linear-attention only reaches here when q_len is uniform.
        layer_cache_group = [runtime_cache_list[idx][local_idx] for idx in range(len(hidden_states_list))]
        batch_hidden_states = self._pad_batch_hidden_states(hidden_states_list, q_len_list, max_q_len)

        if use_continuous_batch_state:
            batch_cache = self._continuous_layer_cache[local_idx]
        else:
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

        merge_mode = "linear merged-continue" if layer.is_linear else "attention merged-batch"
        self.logger.debug(
            "[qwen3_5 decoder] %s layer=%d batch=%d q_lens=%s",
            merge_mode,
            layer_idx,
            len(hidden_states_list),
            q_len_list,
        )
        if layer.is_linear:
            if merge_ssm_mask is None:
                merge_ssm_mask = create_ssm_mask(batch_hidden_states, batch_cache)
            batch_mask = merge_ssm_mask
        else:
            if merge_fa_mask is None:
                merge_fa_mask = create_attention_mask(batch_hidden_states, batch_cache)
            batch_mask = merge_fa_mask
        batch_output = layer(batch_hidden_states, mask=batch_mask, cache=batch_cache)
        if not use_continuous_batch_state and not layer.is_linear:
            batch_cache.finalize()

        for req_idx, q_len in enumerate(q_len_list):
            hidden_states_list[req_idx] = batch_output[req_idx : req_idx + 1, :q_len, :]
            if not use_continuous_batch_state:
                runtime_cache_list[req_idx][local_idx] = batch_cache.extract(req_idx)

        if not use_continuous_batch_state and max_q_len == 1:
            if self._continuous_layer_cache is None:
                self._continuous_layer_cache = [None] * self.local_num_layers
            self._continuous_layer_cache[local_idx] = batch_cache

        return merge_ssm_mask, merge_fa_mask

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
        # Reuse masks in single-path layers within one forward pass.
        ssm_mask_cache = [None] * len(cache.uuid_list)
        fa_mask_cache = [None] * len(cache.uuid_list)
        start = 0
        for uuid in cache.uuid_list:
            q_len = request_cache.get_q_len(uuid)
            end = start + q_len
            hidden_states = h[start:end][None, ...]
            start = end

            hidden_states_list.append(hidden_states)
            runtime_cache_list.append(self._get_request_runtime_cache(uuid))
            q_len_list.append(q_len)
        max_q_len = max(q_len_list)
        right_padding = [max_q_len - q_len for q_len in q_len_list]
        use_continuous_batch_state = self._can_use_continuous_batch_state(cache.uuid_list, q_len_list)
        if not use_continuous_batch_state and self._continuous_layer_cache is not None:
            # On mode switch (e.g. TTFT arrives), write merged caches back once.
            self._materialize_continuous_state(runtime_cache_list)
            self._reset_continuous_state()
        merge_ssm_mask = None
        merge_fa_mask = None

        for local_idx, layer_idx in enumerate(range(self.start_layer_idx, self.end_layer_idx)):
            layer = self.layers[layer_idx]
            layer_cache_group = [runtime_cache_list[idx][local_idx] for idx in range(len(hidden_states_list))]
            if layer.is_linear and len(set(q_len_list)) != 1:
                self._reset_continuous_state()
                self._run_linear_layer_grouped_batch(
                    layer,
                    hidden_states_list,
                    runtime_cache_list=runtime_cache_list,
                    local_idx=local_idx,
                    q_len_list=q_len_list,
                    ssm_mask_cache=ssm_mask_cache,
                )
                continue

            if not self._can_merge_layer_batch(q_len_list, layer_cache_group, layer.is_linear):
                self._reset_continuous_state()
                self.logger.debug(
                    "[qwen3_5 decoder] per-request layer=%d is_linear=%s batch=%d q_lens=%s",
                    layer_idx,
                    layer.is_linear,
                    len(hidden_states_list),
                    q_len_list,
                )
                self._run_layer_per_request(
                    layer,
                    hidden_states_list,
                    layer_cache_group=layer_cache_group,
                    request_indices=list(range(len(hidden_states_list))),
                    ssm_mask_cache=ssm_mask_cache,
                    fa_mask_cache=fa_mask_cache,
                )
                continue

            merge_ssm_mask, merge_fa_mask = self._run_merged_layer_batch(
                layer=layer,
                layer_idx=layer_idx,
                local_idx=local_idx,
                hidden_states_list=hidden_states_list,
                runtime_cache_list=runtime_cache_list,
                q_len_list=q_len_list,
                max_q_len=max_q_len,
                right_padding=right_padding,
                use_continuous_batch_state=use_continuous_batch_state,
                merge_ssm_mask=merge_ssm_mask,
                merge_fa_mask=merge_fa_mask,
            )

        if max_q_len == 1 and self._continuous_layer_cache is not None:
            self._continuous_uuid_order = tuple(cache.uuid_list)
        for uuid in cache.uuid_list:
            self._set_request_kv_len(request_cache, uuid)
        return cat_func([x.squeeze(0) for x in hidden_states_list])
