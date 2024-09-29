## together-LLM

跨机推理 LLM 框架

### RoadMap

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
- [ ] Shard Storage

### Performance

在保证通信带宽的前提下，速度应当更快

| PP,TP   | LLaMA3-8B | 34B | LLaMA3-70B |
| ---- | --------- | --- | ---------- |
| 1,1(baseline) |           |     |            |
| 2,1 | |     |            |
| 2,2 | |     |            |
