import requests

if __name__ == "__main__":
    url = "http://localhost:8000/generate"
    data = {"prompt": "Hello, how are you?", "stream": True}
    response = requests.post(url, json=data, stream=True)
    for x in response:
        print(x)

    data = {"prompt": "Hello, how are you?", "stream": False}
    response = requests.post(url, json=data)
    print(response.json())
