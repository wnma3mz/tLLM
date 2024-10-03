import requests
from tqdm import tqdm

if __name__ == "__main__":
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer common"}
    data = {"messages": [{"role": "user", "content": "Hello, how are you?"}], "stream": True}
    # warm-up
    response = requests.post(url, headers=headers, json=data, stream=True)
    for x in response:
        print(x)