from typing import *

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str = "unknown"
    messages: List[Dict[str, str]]
    max_tokens: int = 20
    do_sample: bool = False
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class ChatCompletionResponse(BaseModel):
    token: List[int]
    cost_time: float
    finish_reason: str
    usage: Dict[str, int]
    text: str
