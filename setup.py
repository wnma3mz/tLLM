from pathlib import Path

from setuptools import find_packages, setup

# 基础依赖
root_dir = Path(__file__).parent
with open(root_dir / "requirements" / "base.txt") as fid:
    install_requires = [l.strip() for l in fid.readlines()]

# 平台特定依赖
with open(root_dir / "requirements" / "mlx.txt") as fid:
    mlx_requires = [l.strip() for l in fid.readlines()]

with open(root_dir / "requirements" / "torch.txt") as fid:
    torch_requires = [l.strip() for l in fid.readlines()]

# 可选功能依赖
extras_require = {
    "mlx": mlx_requires,
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
            "tllm.client=tllm.grpc.worker_service.worker_server:main",
        ],
    },
)
