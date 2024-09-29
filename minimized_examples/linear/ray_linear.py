import time
from typing import List

from pydantic import BaseModel
import ray
import torch
import torch.nn as nn

ray.init(ignore_reinit_error=True)


class MLPData(BaseModel):
    x: List[List[float]]


class MLP(nn.Module):
    def __init__(self, config: dict):
        super(MLP, self).__init__()
        self.proj = nn.Linear(config["input_size"], config["output_size"])

    def forward(self, x):
        return self.proj(x)


hidden_size = 4096


# Create a Ray remote function for model inference
@ray.remote
def mlp_forward(x: torch.Tensor):
    s1 = time.time()
    o = model(x).tolist()
    return {"output": o, "cost_time": time.time() - s1}


# Initialize the model and load its state_dict to be shared
model = MLP({"input_size": hidden_size, "output_size": hidden_size})


def main():
    sample_input = torch.rand((1, hidden_size))  # Batch size of 1, hidden_size features

    # for warmup
    s1 = time.time()
    future = mlp_forward.remote(sample_input)
    result = ray.get(future)
    print("Cost(All) Time:", time.time() - s1)
    print("Calculation time", result["cost_time"])

    for _ in range(3):
        s1 = time.time()
        future = mlp_forward.remote(sample_input)
        result = ray.get(future)
        print("Cost(All) Time:", time.time() - s1)
        print("Calculation time", result["cost_time"])


if __name__ == "__main__":
    main()
