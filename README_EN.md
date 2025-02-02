## together-LLM

[English](README_EN.md) | [中文](README.md) 

Cross-Machine Inference LLM Framework

### Begin quickly!

1. Install dependencies

- On macOS (Apple silicon): `pip install -U -e ".[mlx]"`
- Other platforms (NVIDIA): `pip install -e ".[torch]"`

This machine is running: `python3 ./run_engine.py --model_path mlx-community/Llama-3.2-1B-Instruct-4bit`

2. Start HTTP service

- Single machine: `tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit`
- 
- Multiple machines:
- Start a server for a service: `tllm.server --model_path mlx-community/Llama-3.2-1B-Instruct-4bit --hostname $YOUR_IP`
- In another terminal, start the client `tllm.client --hostname http://$YOUR_IP:8022`

3. Test HTTP service

- `python3 benchmarks/run_async_requests.py`

### Support model

- Llama
- Qwen
- Janus Pro: Only supports MacOS platform
  - Text to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type llm`
  - Image to Text: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type mllm`
  - Text to Image: `PYTHONPATH="./" python3 run_janus_pro.py --model_path wnma3mz/Janus-Pro-1B-4bit --message_type image`
- On MacOS, you need to install `pip install mlx-vlm==0.1.12`.
- Flux is currently only supported on MacOS. To use Flux, you will need to install `pip install mflux=0.4.1`.

### Advanced

For multi-machine deployment, the default ports are used for running. If special requirements are needed, the configuration file `examples/config.json` can be modified.


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
  - [X] Prefill-Cache (Token-Level)
  - [ ] PageAttention

### Performance

In Mac Mini M4

|                                      | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `mlx-community/Llama-3.2-1B-Instruct` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` |
| ------------------------------------ | -------------------------------------------- | --------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Mac Mini M4 (16G) (Engine, Baseline) | 98.10 tok/s                                 | 35.45 tok/s                             | 20.68 tok/s                                       | No Memory                                         |
| Mac Mini M4 (16G) (Local)            | 45.36 tok/s                                 | 23.60 tok/s                             | 15.80 tok/s                                       | No Memory                                         |
| Mac Mini M4 (16G) (Server+Client)    | 61.83 tok/s                                 | 34.54 tok/s                             | 14.91 tok/s                                       | No Memory                                         |
| Mac Mini M4 (16G) + M3 Pro (18G)     |                                              | 16.33 tok/s                             | 11.06 tok/s                                       | 5.64 tok/s                                        |

Q: Why Local is slower than Server+Client?

A:

- Local 只有一个进程，启动了 HTTP Serve， Engine 和 Model 都在一个进程中
- Server+Client 是两个进程，Server 中包含了 HTTP Serve 和 Engine，以及 Embedding 和 LM HEAD；Client 中只有 Model

但不清楚，为什么 `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` 这个不大一样，暂时归因到内存压力上。

Q: Mac Mini M4 (16G) + M3 Pro (18G) 这一列速度为什么慢？

A: In an ideal scenario, it would be equivalent to a Mac Mini M4 (16G) (Server+Client), but due to the need for communication, the communication overhead accounts for a significant portion of the total cost. The main issue is that each token generation requires a certain amount of time, even within a local network.
