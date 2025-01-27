from typing import Dict, List, Optional, Tuple


class Node:
    def __init__(self, request_id: str):
        self.children: Dict[int, Node] = {}
        self.is_end = False
        self.path = None
        self.request_id = request_id

    def __repr__(self):
        return f"Node({self.request_id}): path={self.path}; is_end={self.is_end}"


class RadixTree:
    def __init__(self):
        self.root = Node(None)  # 根节点
        self.request_id_map: Dict[str, Node] = {}

    def insert(self, input_ids: List[int], request_id: str):
        node = self.root
        path = []
        for id_ in input_ids:
            if id_ not in node.children:
                node.children[id_] = Node(request_id)
            node = node.children[id_]
            path.append(id_)
            node.path = path[:]
        node.is_end = True
        self.request_id_map[request_id] = node

    def append_to_request(self, input_ids: List[int], request_id: str):
        if request_id not in self.request_id_map:
            self.insert(input_ids, request_id)
            return
        node = self.request_id_map.pop(request_id)
        path = node.path
        node.is_end = False
        for id_ in input_ids:
            if id_ not in node.children:
                node.children[id_] = Node(request_id)
            node = node.children[id_]
            path.append(id_)
            node.path = path[:]
        node.is_end = True
        self.request_id_map[request_id] = node

    def longest_common_prefix(self, input_ids: List[int]) -> Tuple[Optional[str], int]:
        # 返回最长的公共前缀
        node = self.root
        longest = []
        for id_ in input_ids:
            if id_ not in node.children:
                return node.request_id, len(longest) - 1 if len(longest) > 0 else -1
            node = node.children[id_]
            if node.path is not None and len(node.path) > len(longest):
                longest = node.path[:]
        return node.request_id, len(longest) - 1 if len(longest) > 0 else -1

    def remove(self, input_ids: List[int]):
        # 删除节点
        node = self.root
        for id_ in input_ids:
            if id_ not in node.children:
                return
            node = node.children[id_]
        node.is_end = False
        if len(node.children) == 0:
            del self.request_id_map[node.request_id]
            return
        node.request_id = None
