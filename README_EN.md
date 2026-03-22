## Together-LLM

[English](README_EN.md) | [中文](README.md) 

Cross-Machine Inference LLM Framework

### Quick Start

1. Install dependencies

- Install uv (if needed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install dependencies (MLX only): `uv sync --extra mlx`

This machine is running: `uv run python ./run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit`

2. Start HTTP service

- Single machine: `uv run tllm.server --model_path mlx-community/Qwen3.5-0.8B-4bit`

- Multi-machine:
  - Start a server in a terminal: `uv run tllm.server --model_path mlx-community/Qwen3.5-0.8B-4bit --hostname $YOUR_IP`
  - Start a client on another terminal `uv run tllm.client --hostname http://$YOUR_IP:8022`

3. Test HTTP service

- `uv run python benchmarks/run_async_requests.py`

### Support model

- Qwen3.5 (MLX)
  - Text: `uv run python run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit --message_type llm`
  - VLM: `uv run python run_engine.py --model_path mlx-community/Qwen3.5-0.8B-4bit --message_type vlm`

### Advanced

For multi-machine deployment, the default part of the port will be used for running. If there are special requirements, you can modify it through the configuration file `examples/config.json`.

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

- The number of clients will determine the number of model splits.
- `server.grpc_port`: server's grpc port, used for each client to send status data and the last client to send the computed result
- `server.http_port`: server's http port, API interface as well as WebSocket service
- `server.hostname`: server's hostname, can be replaced with IP, such as 192.168.1.10, make sure client can access
- `client.grpc_port`: client's grpc port
- `client.hostname`: client's hostname, ensure server and other client can access

### Features

- [X] Support Multi-Requests
- [X] Engine
  - [X] mlx
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

|                                                  | `mlx-community/Qwen3.5-0.8B-4bit` | `mlx-community/Qwen3.5-9B-4bit` | `mlx-community/Qwen3.5-35A3-4bit` |
| ------------------------------------------------ | --------------------------------- | ------------------------------- | ---------------------------------- |
| Mac Mini M4 (16G) (Local)                        | 89.73 tok/s                       | 16.22 tok/s                     | -                                  |
| Mac Mini M4 (16G) + M3 Pro (18G) by Thunderbolt5 | -                                 | -                               | -                                  |
| Mac Mini M4 (16G) + M3 Pro (18G) by LAN          | -                                 | -                               | -                                  |
