from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


@dataclass
class TokenizerResult:
    input_ids: List[int]
    input_str: str


class TokenizerUtils:
    def __init__(self, tok_path: Optional[str] = None, tokenizer: Optional[AutoTokenizer] = None):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            assert tok_path is not None, "Either tok_path or tokenizer must be provided."
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=True)

    def preprocess(
        self, text: str = None, messages: List[Dict[str, str]] = None, add_generation_prompt: bool = True
    ) -> TokenizerResult:
        if messages:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        assert text is not None, "Either text or messages must be provided."
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return TokenizerResult(input_ids=input_ids, input_str=text)

    def preprocess_old(self, text: str = None, messages: List[List[Dict[str, str]]] = None) -> TokenizerResult:
        formatted_prompt = "### Human: {}### Assistant:"

        if messages:
            text = formatted_prompt.format(messages[0]["content"])
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        while input_ids[0] == input_ids[1] == self.tokenizer.bos_token_id:
            input_ids.pop(0)
        return TokenizerResult(input_ids=input_ids, input_str=text)

    def decode(
        self, token_ids: List[int], cache_token_ids: List[Optional[List[int]]]
    ) -> Tuple[List[str], List[Optional[List[int]]]]:
        for i, cache_token_id in enumerate(cache_token_ids):
            if cache_token_id is not None:
                token_ids[i] = cache_token_id + [token_ids[i]]

        decode_str_list = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        for i, (token_id, decode_str) in enumerate(zip(token_ids, decode_str_list)):
            # BPE 解码失败，返回 token_ids, 提供给下次解码
            if decode_str.endswith("�"):
                cache_token_ids[i] = token_id if isinstance(token_id, list) else [token_id]
                decode_str_list[i] = ""
            else:
                cache_token_ids[i] = None
        return decode_str_list, cache_token_ids
