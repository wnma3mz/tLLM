## together-LLM

跨机推理 LLM 框架

### RoadMap

使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行，仅通信 hidden_states

- [x] pipeline-parallel by grpc
- [x] tensor-parallel by torch.dist
- [x] Merge Linear
- [ ] Performance Testing
    - [ ] CPU llama-1B
    - [ ] GPU llama-1B
    - [ ] CPU llama-8B
- [ ] Async Generation
    - [ ] Queuing mechanism
- [ ] Decoding Strategy
- [ ] Model
    - [x] LLaMA
- [ ] Multi-Model
    - [ ] LLaVA
- [ ] Web UI
    - [ ] Node Status
- [ ] KV Token Cache
- [ ] MLX Framework
    - [ ] LORA Training
- [ ] Shard Storage
- [ ] split model before load
- [ ] Streaming Output

### Performance


- 2 GHz 四核Intel Core i5, 16 GB 3733 MHz LPDDR4X
- Apple M3 Pro, 18 GB

在保证通信带宽的前提下，速度应当更快

| PP,TP   | TinyLlama-1.1B-Chat-v1.0 | 34B | LLaMA3-70B |
| ---- | --------- | --- | ---------- |
| 1,1(baseline) |    6.37 token/s; 17.98 token/s      |     |            |
| 2,1(单机模拟) | 5.91 token/s|     |            |
| 2,2(单机模拟) | 5.46 token/s |     |            |
