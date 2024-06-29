import socket
from typing import List, Optional

import torch


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def create_decoder_attention_mask(size: int) -> torch.Tensor:
    # Create a lower triangular matrix with ones below the diagonal
    mask = torch.tril(torch.ones(size, size)).transpose(0, 1)
    # Fill the diagonal with ones as well
    mask = mask.masked_fill(mask == 0, float("-inf"))
    return mask


def tensor_to_list(tensor: Optional[torch.Tensor]) -> List:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.float().cpu().detach().numpy().tolist()


def list_to_tensor(lst: Optional[List]) -> torch.Tensor:
    if lst is None:
        return None
    if not isinstance(lst, list):
        return lst
    return torch.tensor(lst)
