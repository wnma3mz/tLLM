import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import websockets


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

    def _create_event_loop(self):
        """创建新的事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        return loop

    async def connect(self):
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
            self.logger.info(f"Connection failed: {e}")
            return False

    async def _run_async(self):
        """异步运行客户端"""
        if not await self.connect():
            return

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
