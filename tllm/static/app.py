import json
import random
from typing import *

import gradio as gr
import requests

url = "http://localhost:8000/v1/chat/completions"


def requests_func(messages: List[Dict[str, Any]]):
    data = {"messages": messages, "model": "tt", "stream": True, "max_tokens": 1024}
    return requests.post(url, json=data, stream=True)


def predict(message, history):
    history_openai_format = []
    for msg in history:
        history_openai_format.append(msg)
    history_openai_format.append({"role": "user", "content": message})

    response = requests_func(history_openai_format)

    partial_message = ""
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            try:
                response = chunk.decode("utf-8").split("data: ")[-1].strip()
                if response == "[DONE]":
                    break
                    # yield partial_message
                data_chunk = json.loads(response)
                if data_chunk["choices"][0]["finish_reason"] is not None:
                    break
                    # yield partial_message
            except:
                print("Error decoding chunk", chunk)
            if data_chunk["choices"][0]["delta"]["content"] is not None:
                partial_message = partial_message + data_chunk["choices"][0]["delta"]["content"]
                yield partial_message


if __name__ == "__main__":
    gr.ChatInterface(predict, type="messages").launch()
