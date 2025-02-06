from tllm.commons.cache import AttentionData, Cache, RequestsCache


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

    def post_init(self, is_start_pp: bool, is_end_pp: bool):
        self.is_start_pp = is_start_pp
        self.is_end_pp = is_end_pp

    def build_forward_cache(self, hidden_states, seq_input):
        raise NotImplementedError

    def get_last_hidden_states(self, hidden_states):
        raise NotImplementedError
