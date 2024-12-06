try:
    import mlx.core as mx  # type: ignore

    HAS_MLX = False
except:
    HAS_MLX = False
