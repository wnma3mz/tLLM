import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from typing import Tuple
import uuid

import requests
from transformers import AutoConfig
import websockets

from tllm.models.register import HAS_MLX, MODEL_REGISTER, load_weight


def get_unregistered_layer_idx(server_url: str) -> Tuple[int, int]:
    # TODO: 获取 server 端未注册的连续的 layer_idx
    response = requests.get(f"{server_url}/unregister_layer_idx")
    # layer_idx = response.json()["data"]
    return -1, -1


class ModelClient:
    def __init__(self, logger, args):
        self.start_idx = args.start_layer_idx
        self.end_idx = args.end_layer_idx
        self.ip_addr = args.ip_addr
        self.port = args.port

        self.client_id = f"client-{str(uuid.uuid4())[:8]}-pp{args.start_layer_idx}-{args.end_layer_idx}"

        self.server_url = args.master_url.replace("http://", "ws://").replace("https://", "wss://")

        self.logger = logger
        self.websocket = None
        self.running = False
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load_model(self, config: AutoConfig, model_path: str, dtype):
        config.decoder_start_layer_idx = self.start_idx
        config.decoder_end_layer_idx = self.end_idx

        arch = config.architectures[0]
        if HAS_MLX:
            arch = "MLX" + arch
        if arch not in MODEL_REGISTER:
            raise ValueError(f"Model {arch} not supported")

        HF_CausalLM_CLASS, _, MY_MODEL_CLASS = MODEL_REGISTER[arch]

        s1 = time.time()
        if HAS_MLX:
            import mlx.core as mx

            weights = load_weight(model_path)

            model = MY_MODEL_CLASS(config)
            for pop_key in ["model.embed_tokens.weight", "model.norm.weight"]:
                if pop_key in weights:
                    weights.pop(pop_key)
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
        else:
            state_dict = HF_CausalLM_CLASS.from_pretrained(
                model_path, trust_remote_code=True, device_map="cpu", torch_dtype=dtype, low_cpu_mem_usage=True
            ).state_dict()
            model = MY_MODEL_CLASS(config).to(dtype)
            model.load_state_dict(state_dict)
            del state_dict

        if hasattr(config, "comm"):
            self.logger.debug(f"[Rank: {config.comm.rank}] Cost time {time.time() - s1}")
        model.eval()
        return model

    def _create_event_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    async def connect(self, cnt: int):
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
                        "ip_addr": self.ip_addr,
                        "port": self.port,
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
