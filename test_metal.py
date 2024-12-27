import mlx.core as mx


def exp_elementwise(a: mx.array):
    source = """
        uint elem = thread_position_in_grid.x;
        T tmp = inp[elem];
        out[elem] = metal::exp(tmp);
    """

    kernel = mx.fast.metal_kernel(
        name="myexp",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )
    outputs = kernel(
        inputs=[a],
        template=[("T", mx.float32)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )
    return outputs[0]


def rotate_half_metal(x: mx.array):
    source = """
        uint elem = thread_position_in_grid.x;
        uint half_dim = size / 2;
        uint batch_idx = elem / size;
        uint dim_idx = elem % size;
        
        if (dim_idx < half_dim) {
            out[elem] = -inp[batch_idx * size + dim_idx + half_dim];
        } else {
            out[elem] = inp[batch_idx * size + dim_idx - half_dim];
        }
    """

    kernel = mx.fast.metal_kernel(
        name="rotate_half",
        input_names=["inp", "size"],
        output_names=["out"],
        source=source,
    )

    last_dim = x.shape[-1]
    total_size = x.size

    outputs = kernel(
        inputs=[x.reshape(-1), last_dim],
        grid=(total_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # Expand dimensions
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_metal(q, k, cos, sin, unsqueeze_dim=1):
    # Expand dimensions
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half_metal(q) * sin)
    k_embed = (k * cos) + (rotate_half_metal(k) * sin)
    return q_embed, k_embed


def apply_rotary_embedding_metal_v2(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    source = """
        uint elem = thread_position_in_grid.x;
        uint size = q_size;
        uint half_dim = size / 2;
        uint batch_idx = elem / size;
        uint dim_idx = elem % size;
        
        // Calculate rotated values
        float rotated_q, rotated_k;
        if (dim_idx < half_dim) {
            rotated_q = -q[batch_idx * size + dim_idx + half_dim];
            rotated_k = -k[batch_idx * size + dim_idx + half_dim];
        } else {
            rotated_q = q[batch_idx * size + dim_idx - half_dim];
            rotated_k = k[batch_idx * size + dim_idx - half_dim];
        }
        
        // Apply rotary embedding
        q_out[elem] = q[elem] * cos[dim_idx] + rotated_q * sin[dim_idx];
        k_out[elem] = k[elem] * cos[dim_idx] + rotated_k * sin[dim_idx];
    """
    cos = mx.expand_dims(cos, axis=-1)
    sin = mx.expand_dims(sin, axis=-1)
    kernel = mx.fast.metal_kernel(
        name="rotary_embed",
        input_names=["q", "k", "cos", "sin", "q_size"],
        output_names=["q_out", "k_out"],
        source=source,
    )

    last_dim = q.shape[-1]
    total_size = q.size

    outputs = kernel(
        inputs=[q, k, cos, sin, last_dim],
        grid=(total_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return outputs[0], outputs[1]


if __name__ == "__main__":
    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))

    q = mx.random.normal(shape=(42, 32, 64)).astype(mx.bfloat16)
    k = mx.random.normal(shape=(42, 8, 64)).astype(mx.bfloat16)
    cos = mx.random.normal(shape=(42, 64)).astype(mx.bfloat16)
    sin = mx.random.normal(shape=(42, 64)).astype(mx.bfloat16)
    # # queries (1, 32, 64)
    # # keys (1, 8, 64)
    # # cos (1, 64)
    # # sin (1, 64)
    from my_ext import apply_rotary_pos_emb as apply_rotary_pos_emb_new

    base_q, base_k = apply_rotary_pos_emb(q, k, cos, sin)
    new_q = apply_rotary_pos_emb_new(q, k, cos, sin)
    new_k = apply_rotary_pos_emb_new(k, q, cos, sin)
    print("assert base", mx.allclose(base_q, new_q))
    print("assert base", mx.allclose(base_k, new_k))
    # b = apply_rotary_pos_emb_metal(q, k, cos, sin)
    # print(a[0], b[0])
    # assert mx.allclose(a[0], b[0])
    # assert mx.allclose(a[1], b[1])
