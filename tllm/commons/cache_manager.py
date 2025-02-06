import copy
from typing import List, Optional

from tllm import ENABLE_PREFILL_CACHE
from tllm.commons.cache import AttentionData, Cache, DecoderCache, RequestsCache, arange_func, array_func


class CacheManager:
    # 管理每个节点的所有层 kv_cache
    # max_alive_time: 超过多久没有访问就删除，单位秒
    def __init__(self, max_alive_time=60):
        self.cache = Cache(max_alive_time)
        self.request_cache: RequestsCache = None
        self.is_start_pp: bool = None
        self.is_end_pp: bool = None
        self.attn_data: AttentionData = None

    def init_request_cache(self, num_layers: int, max_seq_len: int, n_kv_heads: int, head_dim: int):
        self.request_cache = RequestsCache(num_layers, max_seq_len, n_kv_heads, head_dim)

    def contains(self, key) -> bool:
        return self.cache.contains(key)

    def update_cache(self, seq_input):
        for uuid in seq_input.uuid_list:
            self.cache.set(uuid, self.attn_data.request_cache.get_decoder_cache(uuid))
            self.cache.check_alive()

        if self.request_cache is not None:
            self.request_cache.clear()
            self.request_cache.insert_cache(seq_input)

    def _build_single_cache(self, uuid: str, input_ids: List[int], cache: Cache):
        hit_cache_len = -1
        q_len = len(input_ids)
        is_decoding = cache.contains(uuid)

        # decoding 阶段
        if is_decoding:
            decoder_cache: DecoderCache = cache.get(uuid)
            decoder_cache.set_q_len(q_len)
            cache_seq_len = decoder_cache[0].kv_len
            position_ids = array_func(cache_seq_len)
            return q_len, cache_seq_len + q_len, hit_cache_len, position_ids, decoder_cache

        # prefilling 阶段
        if ENABLE_PREFILL_CACHE:
            hit_uuid, hit_cache_len = self.request_cache.radix_tree.longest_common_prefix(input_ids)
        else:
            hit_uuid, hit_cache_len = None, -1

        if hit_uuid is not None and cache.get(hit_uuid) is not None:
            hid_decoder_cache: DecoderCache = copy.deepcopy(cache.get(hit_uuid))
            # 相同输入时，避免过超过 cache 长度
            if q_len <= hit_cache_len:
                hit_cache_len = q_len - 2

            hid_decoder_cache.truncate(hit_cache_len)
            hid_decoder_cache.set_q_len(q_len - hit_cache_len)
            decoder_cache = hid_decoder_cache
            position_ids = arange_func(q_len)
            return q_len, q_len, hit_cache_len, position_ids, decoder_cache
        else:
            hit_cache_len = -1
            decoder_cache = None
            position_ids = arange_func(q_len)
            return q_len, q_len, hit_cache_len, position_ids, decoder_cache

    def build_cache(self, seq_input, cache: Cache):
        q_len_list, k_len_list = [], []
        position_ids_list = []
        hit_cache_len_list = []

        for uuid, input_ids in zip(seq_input.uuid_list, seq_input.input_ids_list):
            q_len, k_len, hit_cache_len, position_ids, decoder_cache = self._build_single_cache(uuid, input_ids, cache)

            q_len_list.append(q_len)
            k_len_list.append(k_len)
            position_ids_list.append(position_ids)
            hit_cache_len_list.append(hit_cache_len)

            self.request_cache.add(uuid, q_len, decoder_cache)
        return q_len_list, k_len_list, position_ids_list, hit_cache_len_list

    def post_init(self, is_start_pp: bool, is_end_pp: bool):
        self.is_start_pp = is_start_pp
        self.is_end_pp = is_end_pp

    def build_forward_cache(self, hidden_states, seq_input):
        raise NotImplementedError

    def get_last_hidden_states(self, hidden_states):
        raise NotImplementedError
