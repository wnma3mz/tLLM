import time
from typing import Dict

import ray
import torch
import torch.nn as nn

torch.set_num_threads(4)
ray.init(ignore_reinit_error=True, num_cpus=4)


@ray.remote
class ParallelLinear(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(row_size, col_size, bias=False)

    def load_state_dict(self, state_dict):
        self.layer.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, x):
        return self.layer(x)


class MyLinear(nn.Module):
    def __init__(self, row_size: int, col_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(row_size, col_size, bias=False)

    def load_state_dict(self, state_dict):
        self.layer.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, x):
        return self.layer(x)


class MyModel(nn.Module):
    def __init__(self, hidden_size: int = 4096, tensor_split: int = 2):
        super(MyModel, self).__init__()
        assert hidden_size % tensor_split == 0, "hidden_size must be divisible by tensor_split"
        self.is_col_layer = False
        if self.is_col_layer:
            self.layer = [ParallelLinear.remote(hidden_size, hidden_size // tensor_split) for _ in range(tensor_split)]
        else:
            self.layer = [ParallelLinear.remote(hidden_size // tensor_split, hidden_size) for _ in range(tensor_split)]
        self.tensor_split = tensor_split

    def load_state_dict(self, state_dict: Dict):
        state_dict_chunks = {}
        for k, v in state_dict.items():
            state_dict_chunks[k] = []
            if self.is_col_layer:
                tensor_list = v.chunk(self.tensor_split, dim=0)
            else:
                tensor_list = v.chunk(self.tensor_split, dim=1)
            for i in range(self.tensor_split):
                state_dict_chunks[k].append(tensor_list[i])

        # Distribute the state_dict chunks to each remote instance
        futures = []
        for idx, layer in enumerate(self.layer):
            data = {k: state_dict_chunks[k][idx] for k in state_dict_chunks}
            futures.append(layer.load_state_dict.remote(data))
        ray.get(futures)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = time.time()
        if self.is_col_layer:
            futures = [self.layer[i].forward.remote(x) for i in range(self.tensor_split)]
            results = ray.get(futures)
            output_tensor = torch.cat(results, dim=-1)
        else:
            split_x_list = torch.chunk(x, self.tensor_split, dim=-1)
            futures = [self.layer[i].forward.remote(split_x_list[i]) for i in range(self.tensor_split)]
            results = ray.get(futures)
            stacked_tensors = torch.stack(results)
            output_tensor = torch.sum(stacked_tensors, dim=0)
        return output_tensor, time.time() - s1


if __name__ == "__main__":
    # Initialize model
    hidden_size, tensor_split = 4096, 1
    model = MyModel(hidden_size=hidden_size, tensor_split=tensor_split)
    print(f"hidden_size: {hidden_size}; tensor_split: {tensor_split}")
    w = torch.randn(hidden_size, hidden_size)
    model.load_state_dict({"weight": w})

    base_model = MyLinear(hidden_size, hidden_size)
    base_model.load_state_dict({"weight": w})

    input_tensor = torch.randn(10, 20, hidden_size)
    output_tensor = model(input_tensor)
    output_tensor = base_model(input_tensor)

    cost_time_list, calc_cost_time_list = [], []
    for _ in range(10):
        s1 = time.time()
        output_tensor, calc_cost_time = model(input_tensor)
        cost_time_list.append(time.time() - s1)
        calc_cost_time_list.append(calc_cost_time)
    print("Cost time:", sum(cost_time_list) / len(cost_time_list))
    print("Calc cost time:", sum(calc_cost_time_list) / len(calc_cost_time_list))
    # print(f"Output tensor: {output_tensor.shape}")

    cost_time_list = []
    for _ in range(10):
        s1 = time.time()
        output_tensor_v2 = base_model(input_tensor)
        cost_time_list.append(time.time() - s1)
    print("Cost time:", sum(cost_time_list) / len(cost_time_list))
    # print(f"Output tensor: {output_tensor.shape}")
