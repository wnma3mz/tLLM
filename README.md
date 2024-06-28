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

#### speed up

- [ ] calculation & communication to be overlapped

### QuickStart

```bash
bash examples/run_rpc_pp2.sh
```

#### model folder

用 `weights`文件夹作为模型存放路径, 会把模型每层拆分出来便于各个客户端读取，例如:

```shell
weights
├── TinyLlama-1.1B-Chat-v1.0.pth
├── layer_0.pth
├── layer_1.pth
├── ......
├── layer_20.pth
├── layer_21.pth
```

### Performance


| PP   | LLaMA3-8B | 34B | LLaMA3-70B |
| ---- | --------- | --- | ---------- |
| base |           |     |            |
| 2    | 1/90 token/s|     |            |
| 4    |           |     |            |
| 8    |           |     |            |
|      |           |     |            |
