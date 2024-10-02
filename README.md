## together-LLM

跨机推理 LLM 框架

### RoadMap

使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行，仅通信 hidden_states

- [x] pipeline-parallel by grpc
- [x] tensor-parallel by torch.dist
- [x] Merge Linear
- [x] Performance Testing
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

由于 tokenizer 可能不同，所以输入 tokens 有一点出入，但基本差不多。

生成 token 速度（减去首token生成的时间）
bfloat 16 CPU
| PP,TP   | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct |
| ---- | --------- | --- | 
| 2,1(实际) | 8.04 token/s | 3.01 token/s |
| 2,2(实际) | 7.38 token/s | 2.51 token/s |

包含首 token 生成时间
| PP,TP   | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct |
| ---- | --------- | --- | 
| 2,1(实际) | 5.49 token/s  | 2.42 token/s  |
| 2,2(实际) | 5.53 token/s  | 2.46 token/s  |


Llama-3.2-1B-Instruct 单机时间：10.96 token/s
Llama-3.2-1B-Instruct 单机时间：5.73 token/s（包含首token生成的时间, transformers 框架 TTFT 时间不方便记录）

TODO: Meta-Llama-3-8B-Instruct in GPU