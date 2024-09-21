## together-LLM

跨机推理 LLM 框架

### RoadMap

#### backend

- [X] transformers
- [X] torch

#### communication

- [X] ray
- [X] torch.dist

http 耗时明显大于 rpc

ray 框架更易使用，无需手写 rpc 相关内容，且通信差不多，底层也是 rpc。

但 ray 框架在使用 Tensor Parallel 会比不开 TP 更快，故暂时先用 torch.dist，可以实现加速的效果。

torch.dist 在 CPU 机器上用 gloo（跨机或许也是用这个？）。

同时，使用 torch 的多进程共享内存进行 TP 也会更慢。

主要注意的是，reduce 涉及到一些复杂的规约算法，会有一些精度问题。

#### parallel strategy

- [ ] pipeline-parallel
- [ ] tensor-parallel

### Performance

#### Pipeline Parallel (PP)

随着 PP 数变大，通信时间增大，所以单 token 时间变大

#### Tensor Parallel (TP)

在保证通信带宽的前提下，速度应当更快

| TP   | LLaMA3-8B | 34B | LLaMA3-70B |
| ---- | --------- | --- | ---------- |
| base |           |     |            |
| 2    | 1 token/ 10s|     |            |
| 4    |           |     |            |
| 8    |           |     |            |
|      |           |     |            |
