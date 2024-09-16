import torch
import torch.nn as nn
import ray
import time

ray.init(ignore_reinit_error=True)


@ray.remote
class ParallelLinear(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(row_size, col_size, bias=False)

    def forward(self, x):
        return self.layer(x)


class MyModel(nn.Module):
    def __init__(self, hidden_size: int = 4096, tensor_split: int = 2):
        super(MyModel, self).__init__()
        assert (
            hidden_size % tensor_split == 0
        ), "hidden_size must be divisible by tensor_split"
        # self.row_layer = [ParallelLinear.remote(hidden_size // tensor_split, hidden_size) for _ in range(tensor_split)]
        self.col_layer = [
            ParallelLinear.remote(hidden_size, hidden_size // tensor_split)
            for _ in range(tensor_split)
        ]
        self.tensor_split = tensor_split

    def forward(self, x):
        futures = []
        s1 = time.time()
        # For row layer
        # for i in range(self.tensor_split):
        #     futures.append(self.row_layer[i].forward.remote(x[:, i * (hidden_size // self.tensor_split):(i + 1) * (hidden_size // self.tensor_split)]))
        # results = ray.get(futures)
        # stacked_tensors = torch.stack(results)
        # output_tensor = torch.sum(stacked_tensors, dim=0)

        # For col layer
        for i in range(self.tensor_split):
            futures.append(self.col_layer[i].forward.remote(x))
        results = ray.get(futures)
        output_tensor = torch.cat(results, dim=1)
        return output_tensor, time.time() - s1


if __name__ == "__main__":
    # Initialize model
    hidden_size, tensor_split = 4096, 2
    model = MyModel(hidden_size=hidden_size, tensor_split=tensor_split)
    print(f"hidden_size: {hidden_size}; tensor_split: {tensor_split}")

    input_tensor = torch.randn(1, hidden_size)
    output_tensor = model(input_tensor)

    cost_time_list, calc_cost_time_list = [], []
    for _ in range(10):
        s1 = time.time()
        output_tensor, calc_cost_time = model(input_tensor)
        cost_time_list.append(time.time() - s1)
        calc_cost_time_list.append(calc_cost_time)
    print("Cost time:", sum(cost_time_list) / len(cost_time_list))
    print("Calc cost time:", sum(calc_cost_time_list) / len(calc_cost_time_list))
    # print(f"Output tensor: {output_tensor.shape}")
