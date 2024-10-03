from dataclasses import dataclass
from typing import List


@dataclass
class SeqInput:
    uuid_str_list: List[str]
    seq_len_list: List[int]
