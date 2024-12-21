## together-LLM

跨机推理 LLM 框架

### QuickStart

1. download model from: https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-bf16

2. install dependencies

- for mlx:   `pip install -r requirements-mlx.txt`
- for nvidia: `pip install -r requirements-cuda.txt`
- for intel: `pip install -r requirements.txt`

3. run server 

    3.1 (no communication)

    - edit `examples/run_single_server.sh`

    ```bash
    bash examples/run_single_server.sh
    ```

    3.2 (with communication)

    - edit `examples/run_client.sh`

    - edit `examples/run_server.sh`

    ```bash
    # first in one terminal
    bash examples/run_server.sh

    # in another terminal
    bash examples/run_client.sh
    ```

4. testing

```python
python benchmarks/run_async_requests.py
```

### Config

In `examples/config.json`

```json
// 客户端的数量会决定模型拆分的数量
{
    "server": {
        "grpc_port": 25001,         // server 的 grpc 端口，用于每个 client 发送状态数据以及最后一个 client 发送计算后的结果
        "http_port": 8022,          // server 的 http 端口，API 接口 以及 WebSocket 服务
        "hostname": "mac-mini"      // server 的 hostname，可以用 ip 代替，如 192.168.1.10，需要确保 client 能够访问
    },
    "client": [
        {
            "grpc_port": 25002,     // 第一个 client 的 grpc 端口
            "hostname": "m3pro"     // 第一个 client 的 hostname，需要确保 server 和 其他 client 能够访问
        },
        {
            "grpc_port": 25003,     // 第二个 client 的 grpc 端口
            "hostname": "m3"        // 第二个 client 的 hostname，需要确保 server 和 其他 client 能够访问
        }
    ]
}
```

### Features

- [x] Support Multi-Requests
- [x] Engine
    - [x] mlx
    - [x] torch
    - [ ] tinygrad
        - [ ] Multi-Request
        - [ ] Jit
        - [ ] Pipeline
- [x] Communication
    - [x] grpc
    - [x] Auto Find Node
        - [x] Simple Get Ip
        - [x] Test Ping
- [x] Attention
    - [x] xformers
    - [x] flash-attn
    - [ ] PageAttention

### Performance

For 1b

- mac mini m2
![alt text](asserts/image.png)

- m3 pro
![alt text](asserts/image-1.png)

for 8b
- m3 pro (layer=8) + mac mini m2 (layer=24) 
![alt text](asserts/image-2.png)