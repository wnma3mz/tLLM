import asyncio
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Union

from fastapi import WebSocket

from tllm.rpc.manager import ClientRPCManager
from tllm.schemas import ClientData, InitModelRequest, InitModelResponse, RegisterClientRequest, RegisterClientResponse
from tllm.websocket.utils import find_continuous_path, parse_model_size, split_model_layers


class WebsocketManager:
    def __init__(self, total_layers: int, model_name: str):
        self.total_layers = total_layers
        self.model_name = model_name
        self.clients: Dict[str, ClientData] = {}
        self.monitor_websockets: Set[WebSocket] = set()  # 前端页面的websocket连接

        self.connect_clients = []
        self.client_size, self.layer_info = split_model_layers(parse_model_size(model_name), total_layers)
        self.client_info = [[start_idx, end_idx, 0] for start_idx, end_idx in self.layer_info]

    def get_free_layer(self) -> Tuple[int, int, int]:
        # 返回一个未被注册的start idx 和 end idx，如果所有层都被注册了，则随机返回一个
        if self.has_full_model:
            pp_rank = random.choice(0, len(self.layer_info))
            return self.layer_info[pp_rank]
        else:
            for pp_rank, (start_idx, end_idx, count) in enumerate(self.client_info):
                if count == 0:
                    return pp_rank, start_idx, end_idx

    async def register_client(self, request: RegisterClientRequest, model_path: str) -> RegisterClientResponse:
        if request.pp_rank == -1:
            self.clients[request.client_id] = ClientData(client_id=request.client_id, host=request.host)

            pp_rank, start_idx, end_idx = self.get_free_layer()
            return RegisterClientResponse(
                pp_rank=pp_rank,
                start_idx=start_idx,
                end_idx=end_idx,
                model=model_path,
                msg="success",
            )
        else:
            # 二次连接
            self.clients[request.client_id] = ClientData(
                client_id=request.client_id,
                host=request.host,
                pp_rank=request.pp_rank,
                start_idx=request.start_idx,
                end_idx=request.end_idx,
            )
            self.client_info[request.pp_rank][-1] += 1
            return RegisterClientResponse(
                pp_rank=request.pp_rank,
                start_idx=request.start_idx,
                end_idx=request.end_idx,
                msg="success",
            )

    async def init_client(self, request: InitModelRequest) -> InitModelResponse:
        if request.client_id not in self.clients:
            return InitModelResponse(msg="client not found", status=499)
        self.clients[request.client_id].start_idx = request.start_idx
        self.clients[request.client_id].end_idx = request.end_idx
        self.clients[request.client_id].pp_rank = request.pp_rank

        self.client_info[request.pp_rank][-1] += 1
        return InitModelResponse(msg="success", status=200)

    async def unregister_client(self, client_id: str):
        if client_id not in self.clients:
            return
        data = self.clients.pop(client_id)
        if data.pp_rank and data.pp_rank != -1:
            self.client_info[data.pp_rank][-1] -= 1

    @property
    def has_full_model(self) -> bool:
        return len(self.connect_clients) == self.client_size

    def get_state(self) -> dict:
        """与前端同步的数据"""
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "client_info": self.client_info,
            "has_full_model": self.has_full_model,
            "connected_clients": len(self.clients),
        }

    def set_connect_clients(self) -> List[str]:
        x = find_continuous_path(self.clients, self.total_layers)
        self.connect_clients = x if x else []

        self.print_host_list()
        return [x.host for x in self.connect_clients]

    def unset_connect_clients(self, idx_list: List[int]) -> List[str]:
        self.connect_clients = [x for i, x in enumerate(self.connect_clients) if i not in idx_list]

    def print_host_list(self):
        print("route path: ", "->".join([x.host for x in self.connect_clients]))

    def find_connect_clients(self, client_id) -> bool:
        for client in self.clients.values():
            if client.client_id == client_id:
                return True
        return False


class PipelineManager:
    def __init__(self, client_size: int):
        self.task: Optional[asyncio.Task] = None
        self.client_size = client_size
        self.client_manager = ClientRPCManager(self.client_size)
        self.last_check_result: List[int] = []
        self.last_check_time: Optional[float] = None

    def update_url(self, host_list: List[str]):
        self.client_manager.update_url(host_list)

    async def send_config(self, master_url: str):
        async def set_single_config(i: int) -> None:
            url = master_url if i == len(self.connect_clients) - 1 else self.connect_clients[i + 1].host
            await self.client_manager.set_config(i, {"forward_url": url, "master_url": master_url, "pp_rank": i})

        tasks = [set_single_config(i) for i in range(len(self.connect_clients))]
        await asyncio.gather(*tasks)

    async def health_check(self) -> Tuple[int]:
        async def check_single_client(index: int) -> Tuple[int, bool]:
            result = await self.client_manager.health_check(index)
            return (index, result)

        tasks = [check_single_client(i) for i, _ in enumerate(self.connect_clients)]

        # 等待所有任务完成，返回结果列表
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # 检查结果，如果有健康检查失败，返回对应的索引
        self.last_check_time = time.time()
        self.last_check_result = [index for index, is_healthy in results if not is_healthy]
        return self.last_check_result

    async def start_health_check_timer(self, interval: float = 60):
        if self.task and not self.task.done():
            return

        async def check_loop():
            while True:
                result = await self.health_check()
                if len(result) > 0:
                    break
                await asyncio.sleep(interval)

        self.task = asyncio.create_task(check_loop())

    def stop_health_check_timer(self):
        if self.task and not self.task.done():
            self.task.cancel()

    async def get_status(self) -> Dict[str, Union[bool, float, List[int]]]:
        return {
            "last_check_time": self.last_check_time,
            "last_check_result": self.last_check_result,
            "is_running": bool(self.task and not self.task.done()),
        }
