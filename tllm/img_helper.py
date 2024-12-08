import base64
from io import BytesIO

from PIL import Image


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_pil_image(base64_string: str) -> BytesIO:
    img_bytes = base64.b64decode(base64_string)
    return BytesIO(img_bytes)


def resize_image_if_needed(img: Image.Image, max_size=512):
    width, height = img.size

    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)

        new_width = int(width * ratio)
        new_height = int(height * ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img
