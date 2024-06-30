import socket
from typing import List, Optional

import numpy as np


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def tensor_to_list(tensor: Optional[np.ndarray]) -> List:
    if tensor is None:
        return None
    if not isinstance(tensor, np.ndarray):
        return tensor
    return tensor.tolist()


def list_to_tensor(lst: Optional[List]) -> np.ndarray:
    if lst is None:
        return None
    if not isinstance(lst, list):
        return lst
    return np.array(lst)
