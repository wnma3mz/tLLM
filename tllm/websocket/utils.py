from typing import Dict, List, Optional, Tuple

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


def parse_model_size(model_name: str) -> float:
    part_list = model_name.lower().rsplit("/", 1)[-1].split("-")
    model_size = -1
    for part in part_list:
        if part.endswith("b"):
            try:
                model_size = float(part[:-1])
                break
            except:
                pass
    assert model_size > 0, f"Invalid model name: {model_name}"
    return model_size


# 根据 model size 和 层数来划分客户端数量以及每个客户端的层数
def split_model_layers(model_size: float, total_layers: int) -> Tuple[int, List[Tuple[int, int]]]:
    if model_size < 4:
        return 1, [(0, total_layers)]
    elif model_size <= 8:
        each_client_layers = total_layers // 2
        return 2, [(0, each_client_layers), (each_client_layers, total_layers)]
    elif model_size <= 32:
        each_client_layers = total_layers // 4
        return 4, [
            (start_idx, start_idx + each_client_layers) for start_idx in range(0, total_layers, each_client_layers)
        ]
    elif model_size <= 72:
        each_client_layers = total_layers // 8
        return 8, [
            (start_idx, start_idx + each_client_layers) for start_idx in range(0, total_layers, each_client_layers)
        ]
    else:
        raise ValueError(f"Model size {model_size} is too large")
