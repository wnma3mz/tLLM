from setuptools import find_packages, setup

# 基础依赖
install_requires = [
    "aiohttp",
    "fastapi",
    "numpy",
    "requests",
    "tabulate",
    "tqdm",
    "typing_extensions",
    "uvicorn",
    "websockets",
    "pillow",
    "huggingface_hub",
    "gradio",
    "psutil",
    "grpcio==1.68.1",
    "lz4==4.3.3",
    "protobuf==5.28.3",
    "pydantic==2.9.2",
    "transformers==4.46.0",
]

# 平台特定依赖
mlx_requires = ["mlx", "mlx_lm==0.19.2"]

tinygrad_requires = [
    "tinygrad",
]

torch_requires = [
    "vllm",
]

# 可选功能依赖
extras_require = {
    "mlx": mlx_requires,
    # 'tinygrad': tinygrad_requires,
    "torch": torch_requires,
    "all": mlx_requires + torch_requires,  # 全部安装（可能在某些平台上无法使用）
    "dev": [
        "black",
        "isort",
    ],
}

setup(
    name="tllm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9",  # 指定最低 Python 版本要求
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "tllm.server=tllm.entrypoints.api_server:main",
            "tllm.client=tllm.entrypoints.handler.handler:main",
        ],
    },
)
