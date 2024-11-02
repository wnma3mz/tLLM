## together-LLM

跨机推理 LLM 框架

### RoadMap

使用 torch.dist 实现 张量并行，使用 rpc 实现流水并行，仅通信 hidden_states

RoadMap

- [ ] Speed Up
    - [x] Merge Linear
    - [x] Pipeline Parallel by grpc
    - [x] Tensor Parallel by torch.dist
    - [x] Sequence KV Cache
    - [x] Performance Testing
    - [ ] Support Flash Attention
- [x] Decoding Strategy
    - [x] Top-K Sampling
    - [x] Top-P Sampling
    - [x] Temperature Sampling
- [ ] Model
    - [x] LLaMA
- [ ] split model before load
- [x] MLX Framework
    - [x] With Torch Inference
        - [x] Some bugs with multi requests
    - [ ] LoRA Training
- [x] Web UI
    - [x] Node Status
        - [ ] Display Multi Model
    - [x] ChatWeb Demo by Gradio
        - [x] Parameters
        - [x] System
        - [x] Button
- [ ] Multi-Modal
    - [ ] LLaVA
- [x] Backend
    - [x] Client Send Url and Port
    - [ ] Auto Layer Split
        - [x] get free layer idx
        - [ ] calculate layer memory and recommend split
    - [x] Async Generation
        - [x] Multi-Sequence Batch=1
        - [x] Queuing mechanism
        - [x] Continuous Batch
        - [x] Test Cases
        - [ ] Client Disconnect
    - [x] OpenAI API format
        - [x] Streaming Output
        - [x] chat completion(stream)
        - [x] chat completion(non-stream)
        - [x] using anythingLLM
    - [ ] Communication
        - [ ] Communication Time Benchmark (in progress)
- [ ] KV Cache
    - [x] Request/Sequence Cache
    - [x] Custom KV Cache Class
    - [ ] Token-Level Cache
        - [ ] Prefix-tree Cache
- [ ] Shard Storage

### Performance

### 网络要求估算

- PP=8 ，那么通信要求需要$*8$
- 70B 的 hidden_size 是 8192
- 数据是 `bfloat16`，每个 token 的通信参数量为 $1*8192*2=16,384$

在 TPOT 阶段预期速度: 20 token/s -> 0.05s / token
- 假设通信：计算比为 1:4，那么通信时间为 0.01s
    - 即每次通信要在 0.01/8s 完成，即 0.00125s-> 1.25ms
    - 当前实现为双向通信，70B 的 hidden_size 是 8192，就有 $16,384*2=32,768$ bytes.
    - 故要在 0.01/8s 完成，那么网络带宽至少要求 $32,768/0.01*8=26,214,400 bytes/s = 26 Mbps$。
在 TTFT 阶段，即首 token 时间预期 3s，
- 假设通信：计算比为 1:2，那么通信时间为 1s，即每次通信要在 1/8s 完成，即 0.125s -> 125ms
- 假设输入 token 数为 1000，那么通信参数量为 $1000*16,384 = 16,384,000$ bytes
- 1/8s 内完成，通信时间为 $16,384,000/1*8=131,072,000 比特/秒 = 131 Mbps$

优化点：
- ring 通信，加速一倍
- 数据压缩一倍，加速一倍
- 在 TTFT 阶段做 PP overlap，把输入 token 分块传输。


- 2 GHz 四核Intel Core i5, 16 GB 3733 MHz LPDDR4X
    - Llama-3.2-1B-Instruct 单机时间：10.96 token/s
    - Llama-3.2-1B-Instruct 单机时间：5.73 token/s（包含首token生成的时间, transformers 框架 TTFT 时间不方便记录）

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
| 2,2(实际) | 5.66 token/s  | 2.46 token/s  |



TODO: Meta-Llama-3-8B-Instruct in GPU

多维数组实现（float32）: 单机通信在 0.002 s 左右 （seq-len=1）
bytes 实现（float32）: 单机通信在 0.001 s 左右 （seq-len=1）