from typing import Dict, Set, Tuple

from fastapi import WebSocket


class ModelManager:
    def __init__(self, total_layers: int, model_name: str):
        self.total_layers = total_layers
        self.model_name = model_name
        self.clients: Dict[str, Tuple[int, int]] = {}  # client_id -> start_idx, end_idx
        self.websockets: Dict[str, WebSocket] = {}  # client_id -> websocket
        self.monitor_websockets: Set[WebSocket] = set()  # 监控页面的websocket连接
        self.layer_counts = [0 for _ in range(self.total_layers)]

    def add_layer_count(self, start_end_idx: Tuple[int, int]):
        start_idx, end_idx = start_end_idx
        for idx in range(start_idx, end_idx):
            self.layer_counts[idx] += 1

    def delete_layer_count(self, start_end_idx: Tuple[int, int]):
        start_idx, end_idx = start_end_idx
        for idx in range(start_idx, end_idx):
            self.layer_counts[idx] -= 1

    def get_layer_statistics(self) -> Dict[int, int]:
        return {idx: value for idx, value in enumerate(self.layer_counts)}

    def has_full_model(self) -> bool:
        return all(self.layer_counts[i] > 0 for i in range(self.total_layers))

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
