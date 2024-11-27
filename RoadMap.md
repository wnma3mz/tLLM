# RoadMap

使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行

- [ ] Speed Up
    - [x] Merge Linear
    - [x] Pipeline Parallel by grpc
    - [x] Tensor Parallel by torch.dist
    - [x] Sequence KV Cache
    - [x] Performance Testing
    - [x] Attention
        - [x] SDPA
        - [x] xformers
        - [x] flash_attention
- [x] Decoding Strategy
    - [x] Top-K Sampling
    - [x] Top-P Sampling
    - [x] Temperature Sampling
- [ ] Model
    - [ ] LLM
        - [x] LLaMA
        - [x] Qwen2
    - [ ] Multi-Modal
        - [x] Qwen2-VL
- [x] MLX Framework
    - [x] With Torch Inference
        - [x] Some bugs with multi requests
    - [x] Quantization
    - [x] MLX Server
    - [ ] LoRA Training
- [x] Web UI
    - [x] Node Status
        - [ ] Display Multi Model
    - [x] ChatWeb Demo by Gradio
        - [x] Parameters
        - [x] System
        - [x] Button
- [x] Backend
    - [x] OpenAI API format
        - [x] Streaming Output
        - [x] chat completion(stream)
        - [x] chat completion(non-stream)
        - [x] using anythingLLM
    - [x] Client Send Url and Port
    - [ ] Auto Layer Split
        - [x] get free layer idx
        - [ ] calculate layer memory and recommend split
        - [ ] split model before load
    - [x] Async Generation
        - [x] Multi-Sequence Batch=1
        - [x] Queuing mechanism
        - [x] Continuous Batch
        - [x] Test Cases
        - [x] Client Disconnect and Abort
        - [x] await Event
    - [x] Communication
        - [x] Communication Time Benchmark
        - [x] Async GRPC
        - [x] Ring Communication
    - [ ] Auto Find Node
        - [x] WebSocket Communication
        - [x] Client Retry Connect
        - [x] Client auto update url 
        - [ ] Master Exit
- [ ] KV Cache
    - [x] Request/Sequence Cache
    - [x] Custom KV Cache Class
    - [ ] Conversation KV Cache (in progress)
    - [ ] Token-Level Cache
        - [ ] Prefix-tree Cache
- [ ] Shard Storage
- [x] Auto Download


Master 和 Client 交互方式 http
- Master 先启动，已知模型名和层数
    - Client 启动 grpc，HTTP 发送可连接到地址信息（TODO 内存/显存大小/算力等信息）到 Master
    - Master 返回模型名，分配的起始和结束层数（同步操作，不需要状态）
    - Client 下载模型，加载模型，向 Master 发送 InitModel 信息完成

    - 之后 Master 会向 Client 定时发送心跳包，确保 Client 连接正常
- 如果 Master 重启，Master 会丢失所有的 Client 信息
    - Client 会有定时心跳检查，带着已有状态重新连接

TODO: master 定时向 client 发送心跳