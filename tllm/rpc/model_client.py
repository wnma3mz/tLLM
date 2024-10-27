import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import websockets

from tllm.models.register import MODEL_REGISTER


class ModelClient:
    def __init__(self, logger, server_url: str, start_idx: int, end_idx: int, client_id: str):
        self.server_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
        self.client_id = client_id
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.logger = logger
        self.websocket = None
        self.running = False
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self, config, model_path: str, dtype):
        config.decoder_start_layer_idx = self.start_idx
        config.decoder_end_layer_idx = self.end_idx

        arch = config.architectures[0]
        if arch not in MODEL_REGISTER:
            raise ValueError(f"Model {arch} not supported")

        HF_CausalLM_CLASS, _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

        s1 = time.time()
        state_dict = HF_CausalLM_CLASS.from_pretrained(
            model_path, trust_remote_code=True, device_map="cpu", torch_dtype=dtype, low_cpu_mem_usage=True
        ).state_dict()
        model = MY_MODEL_CLASS(config).to(dtype)
        model.load_state_dict(state_dict)
        self.logger.info(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
        model.eval()
        del state_dict
        return model

    def _create_event_loop(self):
        """创建新的事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    async def connect(self, cnt: int):
        """建立WebSocket连接"""
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/client/{self.client_id}")
            self.logger.info(f"Connected to server with client_id: {self.client_id}")

            # 注册层信息
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "register_layers",
                        "start_idx": self.start_idx,
                        "end_idx": self.end_idx,
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
            await asyncio.sleep(1)
            try_cnt += 1
            continue

        try:
            while self.running:
                try:
                    # 接收服务器消息
                    message = await self.websocket.recv()
                    self.logger.info(f"Received update: {message}")
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
        """同步方法：启动客户端"""
        if self._thread is not None and self._thread.is_alive():
            self.logger.info("Client is already running")
            return

        self._thread = threading.Thread(target=self._run_in_thread)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """同步方法：停止客户端"""

        def _stop():
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._stop_async(), self._loop)

        self._executor.submit(_stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _stop_async(self):
        """异步停止方法"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
