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
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            attn_mask: Dict[str, Union[torch.Tensor, int]],
        ) -> torch.Tensor:
            """FlashAttention with variable length support"""
            return flash_attn_varlen_func(
                q=query_states.transpose(0, 1),
                k=key_states.transpose(0, 1),
                v=value_states.transpose(0, 1),
                cu_seqlens_q=attn_mask["cu_seqlens_q"],
                cu_seqlens_k=attn_mask["cu_seqlens_k"],
                max_seqlen_q=attn_mask["max_seqlen_q"],
                max_seqlen_k=attn_mask["max_seqlen_k"],
                causal=True,
            ).transpose(0, 1)

        return flash_attention, "flash_attention"

    except ImportError:
        try:
            from xformers.components.attention.core import scaled_dot_product_attention as xformers_attention

            def xformers_attn(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                attn_mask: Optional[torch.Tensor],
            ) -> torch.Tensor:
                """XFormers attention implementation"""
                return xformers_attention(query_states, key_states, value_states, att_mask=attn_mask)

            return xformers_attn, "xformers"

        except ImportError:

            def torch_attn(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                attn_mask: Optional[torch.Tensor],
            ) -> torch.Tensor:
                """PyTorch native attention implementation"""
                return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)

            return torch_attn, "torch"
