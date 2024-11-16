from typing import Dict, List, Optional, Set, Tuple, Union

from fastapi import WebSocket


def find_continuous_path(clients: Dict[str, Dict[str, Union[str, int]]], end_idx: int) -> Optional[List[str]]:
    # 创建一个映射，记录每个start_layer_idx对应的id和end_layer_idx
    layer_map = {item["start_idx"]: item for item in clients.values()}

    # 找到start_layer_idx为0的起始点
    if 0 not in layer_map:
        return None

    path = []
    current_layer = 0

    while current_layer in layer_map:
        current_item = layer_map[current_layer]
        path.append(current_item["client_id"])

        # 如果达到了目标，返回路径
        if end_idx == current_item["start_idx"]:
            return path

        # 更新当前层为下一个起始层
        current_layer = current_item["end_idx"]

        if end_idx == current_layer:
            return path
    return None


def get_url(data: Dict[str, Union[str, int]]) -> str:
    return f"{data['ip_addr']}:{data['port']}"


class WebsocketManager:
    def __init__(self, total_layers: int, model_name: str, master_url: str):
        self.total_layers = total_layers
        self.model_name = model_name
        self.clients: Dict[str, Union[str, int]] = {}  # client_id -> {client_id, start_layer_idx, end_layer_idx}
        self.monitor_websockets: Set[WebSocket] = set()  # 监控页面的websocket连接
        self.layer_counts = [0 for _ in range(self.total_layers)]
        self.ws_clients: Dict[str, WebSocket] = {}  # client_id -> WebSocket
        self.master_url = master_url
        self.url_list = None
        self.client_id_list = None

    def update_pp_url_list(self):
        client_id_list = find_continuous_path(self.clients, self.total_layers)
        if client_id_list is not None:
            # 有序的client_id列表
            self.url_list = [get_url(self.clients[client_id]) for client_id in client_id_list]
            self.client_id_list = client_id_list
        else:
            self.url_list = None
            self.client_id_list = None

    def register_client(self, client_id: str, data: Dict, websocket: WebSocket):
        self.ws_clients[client_id] = websocket
        self.clients[client_id] = {"client_id": client_id}
        self.clients[client_id].update(data)

        for idx in range(data["start_idx"], data["end_idx"]):
            self.layer_counts[idx] += 1

    def unregister_client(self, client_id: str) -> int:
        if client_id not in self.clients:
            return
        self.ws_clients.pop(client_id)
        data = self.clients.pop(client_id)
        for idx in range(data["start_idx"], data["end_idx"]):
            self.layer_counts[idx] -= 1

    def get_layer_statistics(self) -> Dict[int, int]:
        return {idx: value for idx, value in enumerate(self.layer_counts)}

    def has_full_model(self) -> bool:
        return all(self.layer_counts[i] > 0 for i in range(self.total_layers))

    def unregister_layer_idx(self) -> List[int]:
        # 计算哪些层还没有被占用
        return [idx for idx, count in enumerate(self.layer_counts) if count == 0]

    def get_state(self) -> dict:
        """与前端同步的数据"""
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "has_full_model": self.has_full_model(),
            "connected_clients": len(self.clients),
            "layer_statistics": self.get_layer_statistics(),
        }

    async def broadcast_state(self):
        """向所有监控页面广播状态更新"""
        state = self.get_state()
        disconnected = set()

        for ws in self.monitor_websockets:
            try:
                await ws.send_json(state)
            except:
                disconnected.add(ws)

        self.monitor_websockets -= disconnected
