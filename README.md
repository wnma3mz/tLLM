## Together-LLM

[English](README_EN.md) | [中文](README.md) 

跨机推理 LLM 框架

### 快速开始

1. 安装依赖

- 在 MacOS （Apple silicon）:  `pip install -U -e ".[mlx]"`
- 其他平台（NVIDIA）: `pip install -e ".[torch]"`

本机运行：`PYTHONPATH="./" python3 ./run_engine.py --model_path mlx-community/Llama-3.2-1B-Instruct-4bit`

2. 启动 HTTP 服务

- 单机: `tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit`

- 多机:
  - 在一个终端启动服务端: `tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit --hostname $YOUR_IP`
  - 在另一个终端启动客户端 `tllm.client --hostname http://$YOUR_IP:8022`

3. 测试 HTTP 服务

- `python3 benchmarks/run_async_requests.py`

### 支持模型

- llama
- qwen
- qwen-vl
- janus_pro: 暂只支持 MacOS 平台
  - Text to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type llm`
  - Image to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type mllm`
  - Text to Image: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type image`
- flux: 暂只支持 MacOS 平台，需要额外安装 `pip install mflux=0.4.1`


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
- `server.grpc_port`: server 的 grpc 端口，用于每个 client 发送状态数据以及最后一个 client 发送计算后的结果
- `server.http_port`: server 的 http 端口，API 接口 以及 WebSocket 服务
- `server.hostname`: server 的 hostname，可以用 ip 代替，如 192.168.1.10，需要确保 client 能够访问
- `client.grpc_port`: client 的 grpc 端口
- `client.hostname`: client 的 hostname，需要确保 server 和 其他 client 能够访问

### Features

- [X] Support Multi-Requests
- [X] Engine
  - [X] mlx
  - [X] torch
- [X] Communication
  - [X] grpc
  - [X] Auto Find Node
    - [X] Simple Get Ip
    - [X] Test Ping
- [X] Attention
  - [X] xformers
  - [X] flash-attn
  - [X] Prefill-Cache (Token-Level)
  - [ ] PageAttention

### Performance

|                                                  | `Qwen3-0.6B-4bit` | `Qwen3-8B-4bit` | `Qwen3-30B-A3B-4bit` |
| ------------------------------------------------ | ------------------| ----------------| ---------------------|
| Mac Mini M4 (16G) (Local)                        | 113.57 tok/s      | 19.31 tok/s     | No Memory            |
| Mac Mini M4 (16G) + M3 Pro (18G) by Thunderbolt5 | -                 | 13.26 tok/s     | 18.66 tok/s          |
| Mac Mini M4 (16G) + M3 Pro (18G) by LAN          | -                 | 11.34 tok/s     | Failed               |