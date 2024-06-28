from typing import Dict, List

from transformers import AutoConfig, AutoTokenizer


class TokenizerUtils:
    def __init__(self, tok_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=False)

    def preprocess(self, text: str = None, message: List[Dict[str, str]] = None) -> List[int]:
        if message:
            text = self.tokenizer.apply_chat_template(message)
        if text:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            return input_ids
        raise ValueError("Please provide text or message")

    def decode(self, input_ids: List[int]) -> str:
        return self.tokenizer.decode(input_ids)
