from typing import Optional

from pydantic import BaseModel


class ImageGenerationConfig(BaseModel):
    num_inference_steps: int
    height: int
    width: int
    guidance: float = 3.5
    init_image_path: Optional[str] = None
    init_image_strength: Optional[float] = None


class Text2ImageRequest(BaseModel):
    model: str
    prompt: str
    config: ImageGenerationConfig
    seed: Optional[int] = 42
    stream: Optional[bool] = False


class Text2ImageResponse(BaseModel):
    image_url: Optional[str] = None
    base64: Optional[str] = None
