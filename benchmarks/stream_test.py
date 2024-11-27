import time
from tllm.web.app import MessageProcessor, process_response_chunk
import requests

metric_text = "Speed: {speed:.2f} tokens/second"

def test_stream(response):
    tokens_generated = 0
    tokens_per_second = 0.
    start_time = time.time()
    partial_message = ""

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            data_chunk = process_response_chunk(chunk)
            if data_chunk is None:
                break
            
            delta_content = data_chunk["choices"][0]["delta"]["content"]
            if delta_content is not None:
                partial_message += delta_content
                # history[-1][1] = partial_message

                tokens_generated += 1
                current_time = time.time()
                time_elapsed = current_time - start_time
                if tokens_generated == 1:
                    start_time = time.time()
                elif tokens_generated == 0:
                    ttft = time_elapsed

                if tokens_generated > 1:
                    tokens_per_second = (tokens_generated-1) / time_elapsed

                print(metric_text.format(speed=tokens_per_second), end="\r")
                # yield history, self.metric_text.format(ttft_time=ttft, token_nums=tokens_generated, speed=tokens_per_second)
    print("\n")
    print(f"Tokens Generated: {tokens_generated}")




if __name__ == "__main__":
    message_processor = MessageProcessor(model="test")
    message = [
        ["Hello, how are you?", None],
    ]
    url = "http://localhost:8022/v1/chat/completions"


    data = message_processor.prepare_request_data(None, message, system_prompt=None, temperature=0, top_p=-1, top_k=1, max_tokens=200)
    response = requests.post(url, json=data, stream=True)
    test_stream(response)

