from typing import Dict, List, Optional

from tllm.schemas import ClientData


def find_continuous_path(clients: Dict[str, ClientData], end_idx: int) -> Optional[List[ClientData]]:
    # 创建一个映射，记录每个start_layer_idx对应的id和end_layer_idx
    layer_map = {item.start_idx: item for item in clients.values()}

    # 找到start_layer_idx为0的起始点
    if 0 not in layer_map:
        return None

    path = []
    current_layer = 0

    while current_layer in layer_map:
        current_item = layer_map[current_layer]
        path.append(current_item)

        # 如果达到了目标，返回路径
        if end_idx == current_item.start_idx:
            return path

        # 更新当前层为下一个起始层
        current_layer = current_item.end_idx

        if end_idx == current_layer:
            return path
    return None
