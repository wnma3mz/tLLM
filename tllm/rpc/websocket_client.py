import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import threading
import time
from typing import Callable, Optional, Tuple
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


class WebSocketClient:
    def __init__(self, logger, args: HandlerArgs, fetch_interval: float = 100):
        self.handler_args = args
        self.client_id = f"client-{str(uuid.uuid4())[:8]}-pp{args.start_idx}-{args.end_idx}"

        self.server_url = args.master_url.replace("http://", "ws://").replace("https://", "wss://")

        self.logger = logger
        self.websocket = None
        self.running = False
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._latest_data = None
        self._data_lock = threading.Lock()
        self.fetch_interval = fetch_interval

    def load_model(self, comm: SingleNodeCommunicator, model_path: str):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.comm = comm

        config.decoder_start_layer_idx = self.handler_args.start_idx
        config.decoder_end_layer_idx = self.handler_args.end_idx

        if model_path.endswith(".gguf"):
            arch = "MLXLlamaForCausalLM"
        else:
            arch = config.architectures[0]
            if HAS_MLX:
                arch = "MLX" + arch

            if arch not in MODEL_REGISTER:
                raise ValueError(f"Model {arch} not supported")

        _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

        s1 = time.time()
        # if model_path.endswith(".gguf"):
        #     weights, config, _ = load_gguf_weight(model_path)
        #     config.decoder_start_layer_idx = self.handler_args.start_idx
        #     config.decoder_end_layer_idx = self.handler_args.end_idx
        #     config.comm = SingleNodeCommunicator()
        model = MY_MODEL_CLASS.from_pretrained(config, model_path)
        self.logger.debug(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")

        return model

    def _create_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    def process_message(self, message: str):
        """服务器端轮询发送数据，处理接收到的消息"""
        try:
            data = json.loads(message)
            if data["type"] == "forward_url":
                # 获取最新的 forward url
                with self._data_lock:
                    self._latest_data = data
            else:
                self.logger.info(f"Received message: {data}")

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def get_data(self):
        with self._data_lock:
            return self._latest_data

    async def connect(self, cnt: int):
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/client/{self.client_id}")
            self.logger.info(f"Connected to server with client_id: {self.client_id}")

            # 注册层信息
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
            if cnt == 0:
                self.logger.info(f"Connection failed: {e}")
            return False

    async def _run_async(self):
        try_cnt = 0
        while not await self.connect(try_cnt):
            self.logger.debug(f"try cnt: {try_cnt}")
            await asyncio.sleep(1)
            try_cnt += 1
            continue

        try:
            while self.running:
                try:
                    # 接收服务器消息
                    message = await self.websocket.recv()
                    self.logger.info(f"Received update: {message}")
                    self.process_message(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("Connection closed by server")
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
