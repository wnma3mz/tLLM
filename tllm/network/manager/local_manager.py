# coding: utf-8
import math
import time
from typing import List, Tuple

from tllm.commons.communicator import Communicator
from tllm.commons.manager import load_client_model
from tllm.schemas import MIX_TENSOR, SeqInput


class LocalRPCManager:
    # 并不发生通信，直接调用模型
    def __init__(self, model_path: str):
        self.model = load_client_model(0, math.inf, Communicator(), model_path)

    async def forward(self, hidden_states: MIX_TENSOR, seq_input: SeqInput) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, seq_input)
        return output_hidden_states, [time.perf_counter() - s1]

    async def image_forward(
        self,
        hidden_states: MIX_TENSOR,
        text_embeddings: MIX_TENSOR,
        seq_len: int,
        height: int,
        width: int,
        request_id: str,
    ) -> Tuple[MIX_TENSOR, List[float]]:
        s1 = time.perf_counter()
        output_hidden_states = self.model(hidden_states, text_embeddings, seq_len, height, width, [request_id])
        return output_hidden_states, [time.perf_counter() - s1]
