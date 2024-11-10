import mlx.core as mx
from transformers import AutoConfig

from tllm.models.mlx_qwen_vl import Qwen2VisionModel

if __name__ == "__main__":
    model_path = "/Users/jianghulu/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VisionModel(config.vision_config)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model.eval()

    pixel_values = mx.random.uniform(0, 1, (11640, 1176), dtype=mx.bfloat16)
    image_grid_thw = mx.array([[1, 60, 194]])

    out = model(pixel_values, image_grid_thw)
    print("out", out.shape, out.dtype)
    # pixel_values shape:  (11640, 1176)
    # image_grid_thw  [[  1  60 194]]
