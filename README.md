## together-LLM

跨机推理 LLM 框架

### QuickStart

1. install dependencies

- for mlx (macos arm):  `pip install -e ".[mlx]" && pip install -r requirements/mlx.txt`
- for nvidia: `pip install -e ".[torch]"`

2. run server

   2.1 (no communication)

   ```bash
   tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit --hostname localhost --is_local --client_size 1
   ```

   2.2 (with communication)

   ```bash
   # first in one terminal
   tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit --hostname $YOUR_IP

   # in another terminal
   tllm.client --hostname http://$YOUR_IP:8022
   ```
3. testing

```bash
python3 benchmarks/run_async_requests.py
```

### More Details

In `examples/config.json`

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
  - [ ] tinygrad
    - [ ] Multi-Request
    - [ ] Jit
    - [ ] Pipeline
- [X] Communication
  - [X] grpc
  - [X] Auto Find Node
    - [X] Simple Get Ip
    - [X] Test Ping
- [X] Attention
  - [X] xformers
  - [X] flash-attn
  - [x] Prefill-Cache (Token-Level) 
  - [ ] PageAttention

### Performance

In Mac Mini M4

|                      | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `mlx-community/Llama-3.2-1B-Instruct` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` |
| -------------------- | -------------------------------------------- | --------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Mac Mini M4 (16G) (Engine, Baseline)          | 98.10 tok/s                                 | 35.45 tok/s                             | 20.68 tok/s                                       | No Memory |
| Mac Mini M4 (16G) (Local)          | 45.36 tok/s                                 | 23.60 tok/s                             | 15.80 tok/s                                       | No Memory |
| Mac Mini M4 (16G) (Server+Client)           | 61.83 tok/s                                 | 34.54 tok/s                             |  14.91 tok/s                                       | No Memory |
| Mac Mini M4 (16G) + M3 Pro (18G) |                                              | 16.33 tok/s                   | 11.06 tok/s | 5.64 tok/s |

Q: Why Local is slower than Server+Client?

A: 
- Local 只有一个进程，启动了 HTTP Serve， Engine 和 Model 都在一个进程中
- Server+Client 是两个进程，Server 中包含了 HTTP Serve 和 Engine，以及 Embedding 和 LM HEAD；Client 中只有 Model

但不清楚，为什么 `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` 这个不大一样，暂时归因到内存压力上。

Q: Mac Mini M4 (16G) + M3 Pro (18G) 这一列速度为什么慢？

A：理想情况下会等于 Mac Mini M4 (16G) (Server+Client)，但由于需要进行通信，通信开销占了主要部分，其中主要是延迟问题导致每个 token 生成都需要花费一定时间，哪怕在局域网内。