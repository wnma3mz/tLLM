import base64
from io import BytesIO

from PIL import Image
from PIL.ImageFile import ImageFile


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_pil_image(base64_string: str) -> BytesIO:
    img_bytes = base64.b64decode(base64_string)
    return BytesIO(img_bytes)


def resize_image_if_needed(img: ImageFile, max_size=512) -> ImageFile:
    """
    如果图片尺寸超过 max_size，则等比缩小图片。

    Args:
        image_path (str): 输入图片的文件路径。
        max_size (int): 允许的最大尺寸（宽度或高度）。默认为 512。

    Returns:
        PIL.Image.Image or None: 缩小后的图片对象，如果无需缩小则返回 None。
    """
    width, height = img.size

    # 检查是否需要缩小
    if width > max_size or height > max_size:
        # 计算缩放比例
        if width > height:
            # 宽度是长边，以宽度为基准计算新尺寸
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            # 高度是长边或等边，以高度为基准计算新尺寸
            new_height = max_size
            new_width = int(width * (max_size / height))

        # 进行等比缩小
        # 使用高质量的重采样滤波器，例如 LANCZOS
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_img
    return img
