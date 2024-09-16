import requests
import torch
import time

if __name__ == "__main__":
    x = torch.randn(1, 4096).tolist()
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    json_data = {"x": x}
    s1 = time.time()
    s = requests.post("http://localhost:8003/forward", headers=headers, json=json_data)
    print("Cost time:", time.time() - s1)
    print(s.json()["cost_time"])
