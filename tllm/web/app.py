import argparse
import base64
from io import BytesIO
import json
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from PIL import Image
import gradio as gr
import requests

from tllm.static.gradio_data import GenerationConfig, custom_css


def process_response_chunk(chunk: bytes) -> Optional[Dict]:
    try:
        response_text = chunk.decode("utf-8").split("data: ")[-1].strip()
        if response_text == "[DONE]":
            return None
        data_chunk = json.loads(response_text)
        if data_chunk["choices"][0]["finish_reason"] is not None:
            return None
        return data_chunk
    except:
        print("Error decoding chunk", chunk)
        return None


def resize_image_if_needed(img: Image.Image, max_size=512):
    width, height = img.size

    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)

        new_width = int(width * ratio)
        new_height = int(height * ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def pil_image_to_base64(image: Image.Image) -> str:
    image = resize_image_if_needed(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


class MessageProcessor:
    def __init__(self, model: str):
        self.model = model

    def _format_chat_history(
        self, img: Union[str, Image.Image], history: List[List[str]], system_prompt: str
    ) -> List[Dict[str, str]]:
        """将聊天历史转换为OpenAI格式"""
        formatted_history = []

        if system_prompt and len(system_prompt.strip()) > 0:
            formatted_history.append({"role": "system", "content": system_prompt})

        for message in history:
            user_input, assistant_response = message
            if img is None:
                formatted_history.append({"role": "user", "content": user_input})
            else:
                mm_content = [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"base64": pil_image_to_base64(img)}},
                ]
                formatted_history.append({"role": "user", "content": mm_content})
            if assistant_response is not None:
                formatted_history.append({"role": "assistant", "content": assistant_response})

        return formatted_history

    def prepare_request_data(
        self,
        img: Union[str, Image.Image],
        history: List[List[str]],
        system_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "messages": self._format_chat_history(img, history, system_prompt),
            "model": self.model,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }


class ChatInterface:
    def __init__(self):
        self.should_stop = False
        self.config = GenerationConfig()
        self.message_processor = MessageProcessor(model="test")
        self.metric_text = "Tokens Generated: {token_nums}\nSpeed: {speed:.2f} tokens/second"

    def _create_chat_column(self) -> Tuple[gr.Chatbot, gr.Textbox, gr.Button, gr.Button, gr.Button]:
        """创建聊天界面的主列"""
        chatbot = gr.Chatbot(height=600, show_label=False)

        with gr.Row():
            with gr.Column(scale=0.05):
                img = gr.Image(type="pil", label="上传图片", container=True)
            with gr.Column(scale=13):
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="输入消息...",
                        container=False,
                    )
                    submit_btn = gr.Button("发送", elem_classes="button-primary", scale=0.05)

                with gr.Row():
                    stop_btn = gr.Button("停止生成", elem_classes="button-secondary", scale=1)
                    clear_btn = gr.Button("清空对话", elem_classes="button-secondary", scale=1)

        return chatbot, img, msg, submit_btn, stop_btn, clear_btn

    def _create_config_column(self) -> List[gr.components.Component]:
        """创建配置界面的侧列"""
        gr.Markdown("### 模型参数设置")
        components = [
            gr.Textbox(label="url", value=self.config.chat_url, lines=2),
            gr.Textbox(label="System Prompt", value=self.config.system_prompt, lines=3),
            gr.Slider(minimum=0.0, maximum=2.0, value=self.config.temperature, step=0.1, label="Temperature"),
            gr.Slider(minimum=0.0, maximum=1.0, value=self.config.top_p, step=0.1, label="Top P"),
            gr.Slider(minimum=1, maximum=100, value=self.config.top_k, step=1, label="Top K"),
            gr.Slider(minimum=1, maximum=8192, value=self.config.max_tokens, step=64, label="Max Tokens"),
        ]
        return components

    def _handle_bot_response(
        self, img: Union[str, Image.Image], history: List[List[str]], *config: Tuple[str, str, float, float, int, int]
    ) -> Generator:
        self.should_stop = False
        data = self.message_processor.prepare_request_data(img, history, *config[1:])
        chat_url = config[0]
        response = requests.post(chat_url, json=data, stream=True)

        tokens_generated = 0
        tokens_per_second = 0.0
        start_time = time.time()
        partial_message = ""

        for chunk in response.iter_content(chunk_size=1024):
            if self.should_stop:
                break

            if chunk:
                data_chunk = process_response_chunk(chunk)
                if data_chunk is None:
                    break

                delta_content = data_chunk["choices"][0]["delta"]["content"]
                if delta_content is not None:
                    partial_message += delta_content
                    history[-1][1] = partial_message

                    tokens_generated += 1
                    current_time = time.time()
                    time_elapsed = current_time - start_time
                    if tokens_generated == 1:
                        start_time = time.time()

                    if tokens_generated > 1:
                        tokens_per_second = (tokens_generated - 1) / time_elapsed

                    yield history, self.metric_text.format(token_nums=tokens_generated, speed=tokens_per_second)

    def _handle_user_input(self, user_message: str, history: List[List[str]]) -> Tuple[gr.update, List[List[str]]]:
        return gr.update(value="", interactive=True), history + [[user_message, None]]

    def _handle_stop_generation(self) -> None:
        self.should_stop = True

    def _handle_clear_history(self) -> Tuple[List[List[str]], str]:
        self.should_stop = False
        return [], ""

    def create_interface(self) -> gr.Blocks:
        """创建完整的聊天界面"""
        with gr.Blocks(css=custom_css, title="tLLM Chat Demo") as demo:
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot, img, msg, submit_btn, stop_btn, clear_btn = self._create_chat_column()

                with gr.Column(scale=1):
                    config_components = self._create_config_column()
                    metrics = gr.Markdown(value=self.metric_text.format(token_nums=0, speed=0))

            submit_btn.click(self._handle_user_input, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
                self._handle_bot_response, inputs=[img, chatbot, *config_components], outputs=[chatbot, metrics]
            )

            msg.submit(self._handle_user_input, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
                self._handle_bot_response, inputs=[img, chatbot, *config_components], outputs=[chatbot, metrics]
            )

            stop_btn.click(self._handle_stop_generation, queue=False)

            clear_btn.click(self._handle_clear_history, outputs=[chatbot, msg], queue=False)

        return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chat_interface = ChatInterface()
    demo = chat_interface.create_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, show_api=False, prevent_thread_lock=False, share=False)
