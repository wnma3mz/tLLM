## together-LLM

跨机推理 LLM 框架

### RoadMap

#### backend

- [X] transformers
- [ ] numpy?
- [ ] cpp

#### communication

- [X] http
- [X] grpc

#### parallel strategy

- [X] pipeline-parallel
- [X] tensor-parallel

#### base func

- [ ] decode strategy
    - [ ] beam search
    - [x] top-k
    - [x] top-p
    - [x] greedy
    - [x] temperature
    - [ ] logits processor
- [x] detailed log
    - [x] node forward time
    - [x] communication time
    - [x] local forward time

#### speed up

- [ ] calculation & communication to be overlapped

### communication

shape: [batch, seq_len, hidden] 1x10x4096 -> $1*10*4096*4*8$
network: 1 MB/s -> 1 * (10 ** 6)

$\frac{1*10*4096*4*8}{1 * (10 ** 6)}=1.3 s$


### QuickStart

```bash
bash examples/run_rpc_pp2.sh
```

#### model folder

用`weights`文件夹作为模型存放路径, 会把模型每层拆分出来便于各个节点读取。例如:

```shell
weights
├── llama3-8B
    ├── layer_0.pth
    ├── layer_1.pth
    ├── ......
    ├── other.pth
```

每个节点均需存放所有模型参数

- 避免每次加载都需要传输模型
- 每个节点可以灵活控制，可以是位于模型中任何层的任何位置

### Performance


| PP   | LLaMA3-8B | 34B | LLaMA3-70B |
| ---- | --------- | --- | ---------- |
| base |           |     |            |
| 2    | 1 token/ 10s|     |            |
| 4    |           |     |            |
| 8    |           |     |            |
|      |           |     |            |
