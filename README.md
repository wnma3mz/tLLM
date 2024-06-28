## together-LLM

跨机推理 LLM 框架

### RoadMap

#### backend
- [x] transformers 
- [ ] numpy?
- [ ] cpp

#### communication
- [x] http
- [x] grpc

#### parallel strategy
- [x] pipeline-parallel
- [x] tensor-parallel

#### speed up

- [ ] calculation & communication to be overlapped

### QuickStart

```bash
bash examples/run_rpc_pp2.sh
```


#### model folder

用`weights`文件夹作为模型存放路径, 会把模型每层拆分出来便于各个客户端读取，例如:

```shell
weights
├── TinyLlama-1.1B-Chat-v1.0.pth
├── layer_0.pth
├── layer_1.pth
├── layer_10.pth
├── layer_11.pth
├── layer_12.pth
├── layer_13.pth
├── layer_14.pth
├── layer_15.pth
├── layer_16.pth
├── layer_17.pth
├── layer_18.pth
├── layer_19.pth
├── layer_2.pth
├── layer_20.pth
├── layer_21.pth
├── layer_3.pth
├── layer_4.pth
├── layer_5.pth
├── layer_6.pth
├── layer_7.pth
├── layer_8.pth
└── layer_9.pth
```
