import mlx.core as mx
from mlx_lm.models.llama import DynamicNTKScalingRoPE
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding


def get_inv_freq_mx(dim, base, device=None):
    return 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.int32).astype(mx.float32) / dim))


def get_inv_freq_torch(dim, base, device=None):
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))


if __name__ == "__main__":

    inv_mx = get_inv_freq_mx(10, 1000)
    inv_torch = get_inv_freq_torch(10, 1000)
    print("inv_mx", inv_mx)
    print("inv_torch", inv_torch)

    inv_torch_freq_expanded = inv_torch[None, :, None].float().expand(1, -1, 1)[0]
    inv_mx_freq_expanded = mx.expand_dims(inv_mx, (0, 2))[0]
    print("inv_mx_freq_expanded", inv_mx_freq_expanded)
    print("inv_torch_freq_expanded", inv_torch_freq_expanded)

    position_torch_ids_expanded = torch.arange(0, 10)[None, :].float()
    position_mx_ids_expanded = mx.arange(0, 10)[None, :]

    freqs = inv_torch_freq_expanded.float() @ position_torch_ids_expanded.float()  # .transpose(1, 2)
    emb_torch = torch.cat((freqs, freqs), dim=-1)
    print("emb_torch", emb_torch)

    freqs = inv_mx_freq_expanded @ position_mx_ids_expanded
    emb_mx = mx.concatenate((freqs, freqs), axis=-1)
    print("emb_mx", emb_mx)

    # position_ids_expanded = position_ids[None, :].float()
    assert False

    head_dim = 10
    rope_theta = 10000
    rope_scale = 1.0
    max_position_embeddings = 8192
    rope_scaling = {"factor": 1.0, "low_freq_factor": 1.0, "high_freq_factor": 1.0}
    dyn_rope = DynamicNTKScalingRoPE(
        dims=head_dim,
        max_position_embeddings=max_position_embeddings,
        traditional=False,
        base=rope_theta,
        scale=rope_scale,
        rope_type="llama3",
        rope_scaling=rope_scaling,
    )

    print("Dyn Rope", dyn_rope._freqs, dyn_rope._freqs.shape)

    base_rope = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        base=rope_theta,
        scaling_factor=rope_scale,
    )
    print("Base RoPE", base_rope.inv_freq, base_rope.inv_freq.shape)
