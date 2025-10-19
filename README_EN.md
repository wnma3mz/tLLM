## Together-LLM

[English](README_EN.md) | [中文](README.md) 

Cross-Machine Inference LLM Framework

### Quick Start

1. Install dependencies

- On macOS (Apple silicon): `pip install -U -e ".[mlx]"`
- Other platforms (NVIDIA): `pip install -e ".[torch]"`

This machine is running: `python3 ./run_engine.py --model_path mlx-community/Qwen3-0.6B-4bit`

2. Start HTTP service

- Single machine: `tllm.server --model_path mlx-community/Qwen3-0.6B-4bit`

- Multi-machine:
  - Start a server in a terminal: `tllm.server --model_path mlx-community/Qwen3-0.6B-4bit --hostname $YOUR_IP`
  - Start a client on another terminal `tllm.client --hostname http://$YOUR_IP:8022`

3. Test HTTP service

- `python3 benchmarks/run_async_requests.py`

### Support model

- Llama
- Qwen
- Janus Pro: Currently only supports MacOS platform
  - Text to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type llm`
  - Image to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type mllm`
  - Text to Image: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type image`
- Qwen-VL: On MacOS platform, additional installation is required: `pip install mlx-vlm==0.1.17`.
- flux: Currently only supports MacOS platform, requires additional installation `pip install mflux=0.4.1`.

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
| ------------------------------------------------ | ----------------- | -------------- -| -------------------- |
| Mac Mini M4 (16G) (Local)                        | 115.95 tok/s      | 19.31 tok/s     | No Memory            |
| Mac Mini M4 (16G) + M3 Pro (18G) by Thunderbolt5 | -                 | 13.26 tok/s     | 18.66 tok/s          |
| Mac Mini M4 (16G) + M3 Pro (18G) by LAN          | -                 | 11.34 tok/s     | Failed               |

|                                                  | `Qwen3-VL-4B-Instruct-3bit` | `Qwen3-VL-8B-Instruct-3bit` | `Qwen3-VL-30B-A3B-Instruct-3bit`|
| ------------------------------------------------ | --------------------------- | --------------------------- | ------------------------------- |
| Mac Mini M4 (16G) (Local)                        | 31.40 tok/s                 | 22.76 tok/s                 | No Memory                       |
| M3 Pro (18G) + Mac Mini M4 (16G) by Thunderbolt5 | -                           | 18.25 tok/s                 | 28.51 tok/s                     |
