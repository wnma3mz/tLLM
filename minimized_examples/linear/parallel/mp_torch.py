import time

import torch
from torch.multiprocessing import Array, Process
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def compute_linear(model, input_array, output_array, input_size):
    # 从共享内存读取输入
    input_tensor = torch.tensor(list(input_array)).view(-1, input_size)
    with torch.no_grad():
        output_tensor = model(input_tensor)
        # 将输出写入共享内存
        for i in range(output_tensor.numel()):
            output_array[i] = output_tensor.view(-1)[i].item()


def base_compute_linear(models, input_tensor):
    lst = []
    for model in models:
        with torch.no_grad():
            output_tensor = model(input_tensor)
            lst.append(output_tensor)


if __name__ == "__main__":
    tp_size = 2
    bs = 2
    hidden_size = 4096
    models = [LinearModel(hidden_size, hidden_size) for _ in range(tp_size)]

    s1 = time.time()
    base_compute_linear(models, torch.randn(bs, hidden_size))
    print(time.time() - s1)

    # 创建共享内存用于输入和输出
    inputs = Array("f", bs * hidden_size)  # 输入数据共享内存
    outputs = [Array("f", model(torch.randn(bs, hidden_size)).numel()) for model in models]

    # 填充输入数据到共享内存
    # for i in range(bs):
    #     for j in range(hidden_size):
    #         inputs[i * hidden_size + j] = torch.randn(1).item()

    processes = []
    for model, output_array in zip(models, outputs):
        p = Process(target=compute_linear, args=(model, inputs, output_array, hidden_size))
        processes.append(p)
        p.start()

    s1 = time.time()
    for p in processes:
        p.join()
    print(time.time() - s1)
    print(f"output: ", outputs)
    # 从共享内存读取输出
    # for i, output_array in enumerate(outputs):
    #     output_tensor = torch.tensor(output_array).view(num_inputs, -1)  # 根据输出大小重新形状
    #     print(f"Output {i}: {output_tensor.shape}")
