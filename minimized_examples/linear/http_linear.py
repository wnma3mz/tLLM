import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI()


class MLPData(BaseModel):
    x: List[List[float]]


class MLP(nn.Module):
    def __init__(self, config: dict):
        super(MLP, self).__init__()
        self.proj = nn.Linear(config["input_size"], config["output_size"])

    def forward(self, x):
        return self.proj(x)


app.layer = MLP({"input_size": 4096, "output_size": 4096})


@app.post("/forward")
def forward_mlp(mlp_data: MLPData):
    input_x = torch.tensor(mlp_data.x)
    s1 = time.time()
    output = app.layer(input_x)
    return {"output": output.tolist(), "cost_time": time.time() - s1}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
