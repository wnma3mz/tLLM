import requests
from tqdm import tqdm

if __name__ == "__main__":
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer common"}
    data = {"messages": [{"role": "user", "content": "Hello, how are you?"}]}
    # warm-up
    response = requests.post(url, headers=headers, json=data)

    cost_time_list = []
    for _ in tqdm(range(10)):
        response = requests.post(url, headers=headers, json=data)
        cost_time_list.append(response.json()["cost_time"])
    # 去掉最大值和最小值
    cost_time_list = sorted(cost_time_list)[1:-1]
    test_tokens = 20
    print("cost time: ", sum(cost_time_list) / len(cost_time_list))
    print("token/s: ", test_tokens / (sum(cost_time_list) / len(cost_time_list)))
