from dataclasses import dataclass
from typing import Dict, List

from transformers import AutoTokenizer


@dataclass
class TokenizerResult:
    input_ids: List[int]
    input_str: str


class TokenizerUtils:
    def __init__(self, tok_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=True)

    def preprocess(
        self, text: str = None, messages: List[Dict[str, str]] = None, add_generation_prompt: bool = True
    ) -> TokenizerResult:
        if messages:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        assert text is not None, "Either text or messages must be provided."
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return TokenizerResult(input_ids=input_ids, input_str=text)

    def preprocess_old(self, text: str = None, messages: List[List[Dict[str, str]]] = None) -> TokenizerResult:
        formatted_prompt = "### Human: {}### Assistant:"

        if messages:
            text = formatted_prompt.format(messages[0]["content"])
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        while input_ids[0] == input_ids[1] == self.tokenizer.bos_token_id:
            input_ids.pop(0)
        return TokenizerResult(input_ids=input_ids, input_str=text)

    def decode(self, token_ids: List[int]) -> List[str]:
        return self.tokenizer.batch_decode(token_ids)
