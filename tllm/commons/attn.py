from enum import Enum
import importlib.util
import os
from typing import Dict, Optional

from tllm import BACKEND, BackendEnum

if BACKEND == BackendEnum.TORCH:
    import torch
    import torch.nn.functional as F

    class AttnBackendEnum(Enum):
        AUTO = 0
        TORCH = 1
        VLLM = 2
        XFormers = 3

    if os.environ.get("TLLM_ATTN_BACKEND", None):
        ATTN_BACKEND = AttnBackendEnum[os.environ["TLLM_ATTN_BACKEND"]]
    else:
        ATTN_BACKEND = AttnBackendEnum.TORCH

    if ATTN_BACKEND in [AttnBackendEnum.AUTO, AttnBackendEnum.VLLM] and importlib.util.find_spec("vllm"):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        def flash_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            """FlashAttention with variable length support"""
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=attn_mask["cu_seqlens_q"],
                cu_seqlens_k=attn_mask["cu_seqlens_k"],
                max_seqlen_q=attn_mask["max_seqlen_q"],
                max_seqlen_k=attn_mask["max_seqlen_k"],
                causal=True,
            )

        ATTN_FUNC, ATTN_TYPE = flash_attention, "flash_attention"
    elif ATTN_BACKEND in [AttnBackendEnum.AUTO, AttnBackendEnum.XFormers] and importlib.util.find_spec("xformers"):
        from xformers.components.attention.core import scaled_dot_product_attention as xformers_attention
        from xformers.ops import fmha

        def xformers_attn(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]
        ) -> torch.Tensor:
            """XFormers attention implementation"""
            # return fmha.memory_efficient_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_bias=attn_mask)[0]
            return (
                xformers_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), att_mask=attn_mask)
                .transpose(0, 1)
                .contiguous()
            )

        ATTN_FUNC, ATTN_TYPE = xformers_attn, "xformers"
    else:

        def torch_attn(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]
        ) -> torch.Tensor:
            """PyTorch native attention implementation"""
            return (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=attn_mask
                )
                .transpose(0, 1)
                .contiguous()
            )

        ATTN_FUNC, ATTN_TYPE = torch_attn, "torch"
elif BACKEND == BackendEnum.MLX:
    ATTN_FUNC, ATTN_TYPE = None, "mlx"
else:
    raise ValueError("Invalid backend")
