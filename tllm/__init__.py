try:
    import mlx.core as mx  # type: ignore

    HAS_MLX = True
except:
    HAS_MLX = False
