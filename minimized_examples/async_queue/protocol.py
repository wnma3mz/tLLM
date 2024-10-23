import asyncio
from dataclasses import dataclass, field
from typing import *


@dataclass
class SequenceRequestData:
    # 每个请求在输入输出模型的数据
    request_id: str
    input_ids: Optional[List[int]] = None
    finish_reason: Optional[str] = None

    sampling_params: Optional[Dict] = None

    output_ids: Optional[List[int]] = None
    output_text: Optional[str] = None

    ttft_cost_time: Optional[List[float]] = None
    tpot_cost_time: Optional[List[float]] = None
    timeout: int = 100000  # 请求的总超时时间
    is_stop: bool = False

    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def __repr__(self) -> str:
        return f"request_id={self.request_id}; output_ids={self.output_ids}"
