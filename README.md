## Together-LLM

[English](README_EN.md) | [中文](README.md) 

跨机推理 LLM 框架

### 快速开始

1. 安装依赖

- 安装 uv（如果尚未安装）：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 安装依赖（仅 MLX）：`uv sync --extra mlx`

本机运行：`uv run python ./run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit`

2. 启动 HTTP 服务

- 单机: `uv run tllm.server --model_path mlx-community/Qwen3.5-0.8B-4bit`

- 多机:
  - 在一个终端启动服务端: `uv run tllm.server --model_path mlx-community/Qwen3.5-0.8B-4bit --hostname $YOUR_IP`
  - 在另一个终端启动客户端 `uv run tllm.client --hostname http://$YOUR_IP:8022`

3. 测试 HTTP 服务

- `uv run python benchmarks/run_async_requests.py`

### 支持模型

- Qwen3.5（MLX）
  - 文本: `uv run python run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit --message_type llm`
  - 多模态: `uv run python run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit --message_type vlm`


### 进阶功能

对于多机部署，会使用默认的部分端口进行运行。如果有特殊需求，可以通过配置文件 `examples/config.json` 进行修改。

```json
{
    "server": {
        "grpc_port": 25001,
        "http_port": 8022,
        "hostname": "mac-mini"
    },
    "client": [
        {
            "grpc_port": 25002,
            "hostname": "m3pro"
        },
        {
            "grpc_port": 25003,
            "hostname": "m3"
        }
    ]
}
```

- 客户端的数量会决定模型拆分的数量
- `server.grpc_port`: 服务端的 grpc 端口，用于每个客户端发送状态数据以及最后一个客户端发送计算后的结果
- `server.http_port`: 服务端的 http 端口，API 接口以及 WebSocket 服务
- `server.hostname`: 服务端的 hostname，可以用 ip 代替，如 192.168.1.10，需要确保客户端能够访问
- `client.grpc_port`: 客户端的 grpc 端口
- `client.hostname`: 客户端的 hostname，需要确保服务端和其他客户端能够访问

### 功能

- [X] 支持并发请求
- [X] 推理引擎
  - [X] mlx
- [X] 通信
  - [X] grpc
  - [X] 自动发现节点
    - [X] 自动获取 IP
    - [X] 连通性测试
- [X] 注意力
  - [X] xformers
  - [X] flash-attn
  - [X] Prefill 缓存（Token 级）
  - [ ] PageAttention

### 性能测试

|                                                  | `mlx-community/Qwen3.5-0.8B-4bit` | `mlx-community/Qwen3.5-9B-4bit` | `mlx-community/Qwen3.5-35A3-4bit` |
| ------------------------------------------------ | --------------------------------- | ------------------------------- | ---------------------------------- |
| Mac Mini M4 (16G)（本地）                        | 89.73 tok/s                       | 16.22 tok/s                     | -                                  |
| Mac Mini M4 (16G) + M3 Pro (18G)（Thunderbolt5） | -                                 | -                               | -                                  |
| Mac Mini M4 (16G) + M3 Pro (18G)（局域网）       | -                                 | -                               | -                                  |
