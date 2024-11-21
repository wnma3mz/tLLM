from typing import Dict, Optional, Union

import torch


def get_attention_implementation():
    """
    Get the best available attention implementation.
    Returns a tuple of (attention_func, implementation_name)
    """
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        def flash_attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Dict[str, Union[torch.Tensor, int]]
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

        return flash_attention, "flash_attention"

    except ImportError:
        try:
            from xformers.components.attention.core import scaled_dot_product_attention as xformers_attention

            def xformers_attn(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]
            ) -> torch.Tensor:
                """XFormers attention implementation"""
                return xformers_attention(q, k, v, att_mask=attn_mask)

            return xformers_attn, "xformers"

        except ImportError:
            import torch.nn.functional as F

            def torch_attn(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]
            ) -> torch.Tensor:
                """PyTorch native attention implementation"""
                return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            return torch_attn, "torch"
