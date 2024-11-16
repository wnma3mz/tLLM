import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import threading
from typing import Dict, Optional, Tuple
import uuid

import requests
from transformers import AutoConfig
import websockets

from tllm.commons.communicator import SingleNodeCommunicator
from tllm.models.register import HAS_MLX, MODEL_REGISTER


def get_unregistered_layer_idx(server_url: str) -> Tuple[int, int]:
    # TODO: 获取 server 端未注册的连续的 layer_idx
    response = requests.get(f"{server_url}/unregister_layer_idx")
    # layer_idx = response.json()["data"]
    return -1, -1


@dataclass
class HandlerArgs:
    start_idx: int
    end_idx: int
    ip_addr: str
    port: int
    master_url: str


class ModelManager:
    def __init__(self, start_idx: int, end_idx: int):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def load_model(self, comm: SingleNodeCommunicator, model_path: str):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.comm = comm

        config.decoder_start_layer_idx = self.start_idx
        config.decoder_end_layer_idx = self.end_idx

        if model_path.endswith(".gguf"):
            arch = "MLXLlamaForCausalLM"
        else:
            arch = config.architectures[0]
            if HAS_MLX:
                arch = "MLX" + arch

            if arch not in MODEL_REGISTER:
                raise ValueError(f"Model {arch} not supported")

        _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

        # if model_path.endswith(".gguf"):
        #     weights, config, _ = load_gguf_weight(model_path)
        #     config.decoder_start_layer_idx = self.start_idx
        #     config.decoder_end_layer_idx = self.end_idx
        #     config.comm = SingleNodeCommunicator()
        model = MY_MODEL_CLASS.from_pretrained(config, model_path)
        return model


class WebSocketClient:
    def __init__(self, logger, args: HandlerArgs, fetch_interval: float = 100):
        self.handler_args = args
        self.client_id = f"{str(uuid.uuid4())[:8]}-pp{args.start_idx}-{args.end_idx}"

        self.server_url = args.master_url

        self.logger = logger
        self.websocket = None
        self.running = False
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._latest_data = None
        self._data_lock = threading.Lock()
        self.config_updated = asyncio.Event()
        self.update_callbacks = []
        self.fetch_interval = fetch_interval

    def _create_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    def add_update_callback(self, callback):
        """添加配置更新回调函数"""
        self.update_callbacks.append(callback)

    async def process_message(self, message: str):
        """服务器端轮询发送数据，处理接收到的消息"""
        try:
            data = json.loads(message)
            if data["type"] == "forward_url":
                # 获取最新的 forward url
                with self._data_lock:
                    if data != self._latest_data:
                        self._latest_data = data
                        for callback in self.update_callbacks:
                            await callback(data["master_url"], data["forward_url"], data["pp_rank"])
            else:
                self.logger.info(f"Received message: {data}")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def get_config(self) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        with self._data_lock:
            if self._latest_data:
                return self._latest_data["master_url"], self._latest_data["forward_url"], self._latest_data["pp_rank"]
        return None, None, None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/client/{self.client_id}")
            self.logger.info(f"Connected to server with client_id: {self.client_id}")

            # 注册客户端
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "register_layers",
                        "start_idx": self.handler_args.start_idx,
                        "end_idx": self.handler_args.end_idx,
                        "ip_addr": self.handler_args.ip_addr,
                        "port": self.handler_args.port,
                    }
                )
            )

            self.running = True
            return True
        except Exception as e:
            self.logger.info(f"Connection failed: {e}")
            return False

    async def reconnect(self):
        """处理重连逻辑"""
        try_cnt = 0
        while self.running:
            self.logger.info(f"Attempting to reconnect... (attempt {try_cnt + 1})")
            if await self.connect():
                return True

            # 指数退避策略
            wait_time = min(1 * (try_cnt + 1), 30)  # 最大等待30秒
            self.logger.debug(f"Reconnection failed, waiting {wait_time} seconds before next attempt")
            await asyncio.sleep(wait_time)
            try_cnt += 1
        return False

    async def _run_async(self):
        self.running = True

        # 首次连接
        if not await self.reconnect():
            return

        try:
            while self.running:
                try:
                    # 接收服务器消息
                    message = await self.websocket.recv()
                    self.logger.info(f"Received update: {message}")
                    await self.process_message(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("Connection closed by server")
                    if not await self.reconnect():
                        break
                except Exception as e:
                    self.logger.info(f"Error receiving message: {e}")
                    break
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()

    def _run_in_thread(self):
        """在新线程中运行事件循环"""
        loop = self._create_event_loop()
        loop.run_until_complete(self._run_async())
        loop.close()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            self.logger.info("Client is already running")
            return

        self._thread = threading.Thread(target=self._run_in_thread)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        def _stop():
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._stop_async(), self._loop)

        self._executor.submit(_stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _stop_async(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()
