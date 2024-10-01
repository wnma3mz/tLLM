from dataclasses import dataclass
from typing import Dict, List

from transformers import AutoTokenizer


@dataclass
class TokenizerResult:
    input_ids: List[int]
    input_str: str


class TokenizerUtils:
    def __init__(self, tok_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=False)

    def preprocess(self, text: str = None, messages: List[Dict[str, str]] = None) -> TokenizerResult:
        if messages:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return TokenizerResult(input_ids=input_ids, input_str=text)

    def preprocess_old(self, messages: List[List[Dict[str, str]]]) -> TokenizerResult:
        formatted_prompt = "### Human: {}### Assistant:"

        input_str = formatted_prompt.format(messages[0]["content"])
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=True)
        while input_ids[0] == input_ids[1] == self.tokenizer.bos_token_id:
            input_ids.pop(0)
        return TokenizerResult(input_ids=input_ids, input_str=input_str)

    def decode(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(input_ids)
