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