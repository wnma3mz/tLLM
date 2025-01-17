import asyncio
from datetime import datetime
import time
from typing import Callable, List, Optional

import aiohttp

from tllm.schemas import InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse


class HTTPClient:
    def __init__(
        self,
        master_url: str,
        comm,
        logger,
        ping_interval: int = 30,
        max_retry_attempts: int = 100,
        retry_delay: int = 5,
    ):
        self.master_url = master_url
        self.is_running = False
        self.init_model_info = None
        self.last_ping_time: Optional[datetime] = None
        self.ping_interval = ping_interval  # 健康检查的时间间隔
        self.retry_delay = retry_delay  # ping 不通时，重试的时间间隔
        self.max_retry_attempts = max_retry_attempts  # 最大重试机制，重试失败后，抛出异常
        self.logger = logger
        self.comm = comm
        self.has_connected = False

    async def ping(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.master_url}/health") as response:
                    if response.status == 200:
                        self.last_ping_time = datetime.now()
                        return True
                    return False
        except Exception as e:
            self.logger.error(f"Ping failed")
            return False

    async def register_client(self, request_data: RegisterClientRequest):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.master_url}/register_client", json=request_data.dict(), timeout=30
            ) as response:
                return RegisterClientResponse(**await response.json())

    async def init_model(self, request_data: InitModelRequest):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.master_url}/init_model", json=request_data.dict(), timeout=3) as response:
                return InitModelResponse(**await response.json())

    async def maintain_connection(
        self, client_id: str, ip_addr_list: List[str], port: int, load_model_func: Callable, tp_rank: int = -1
    ):
        while self.is_running:
            # 初次连接 or 连接断开时，尝试重连
            retry_count, is_connected = 0, False
            while retry_count < self.max_retry_attempts:
                is_connected = await self.ping()
                if is_connected:
                    break

                self.has_connected = False
                self.logger.warning("Connection lost, attempting to reconnect...")
                retry_count += 1
                await asyncio.sleep(self.retry_delay)

            # 超过最大重试次数，抛出异常
            if not is_connected:
                self.is_running = False
                raise asyncio.CancelledError("Max retry attempts reached, connection lost")

            # ping 通后，进行连接
            if not self.has_connected:
                try:
                    await self.connect(client_id, ip_addr_list, port, load_model_func, tp_rank)
                    if await self.ping():
                        break
                except Exception as e:
                    self.logger.error(f"Connection Failed {str(e)}")

            await asyncio.sleep(self.ping_interval)

    async def connect(
        self, client_id: str, ip_addr_list: List[str], port: int, load_model_func: Callable, tp_rank: int
    ):
        """定期发送连接请求的协程"""
        if not self.init_model_info:
            register_request = RegisterClientRequest(client_id=client_id, host=ip_addr_list, port=port, tp_rank=tp_rank)
            self.logger.info("RegisterClientRequest: " + str(register_request.dict()))
            response: RegisterClientResponse = await self.register_client(register_request)
            if response.start_idx == -1:
                self.logger.error("Connection failed(start_idx == -1)")
                raise Exception("Connection failed")

            s1 = time.perf_counter()
            load_model_func(response.repo_path, response.start_idx, response.end_idx)
            self.logger.info(
                f"Model loaded in {time.perf_counter() - s1:.4f}s: layer={response.start_idx}-{response.end_idx}"
            )

            self.init_model_info = {
                "pp_rank": response.pp_rank,
                "start_idx": response.start_idx,
                "end_idx": response.end_idx,
            }
            init_request = InitModelRequest(client_id=client_id, **self.init_model_info)
            self.init_model_info["tp_rank"] = tp_rank
            response = await self.init_model(init_request)
            self.logger.info(f"Connection successful")
            self.has_connected = True
        else:
            register_request = RegisterClientRequest(
                client_id=client_id,
                host=ip_addr_list,
                port=port,
                tp_rank=self.init_model_info["tp_rank"],
                pp_rank=self.init_model_info["pp_rank"],
                start_idx=self.init_model_info["start_idx"],
                end_idx=self.init_model_info["end_idx"],
            )
            response: RegisterClientResponse = await self.register_client(register_request)
            if response.start_idx == -1:
                self.logger.error(f"Connection failed")
                raise Exception("Connection failed")
            self.has_connected = True
