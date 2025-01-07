from tllm import BACKEND, BackendEnum
from tllm.entrypoints.handler.master_handler import MasterHandler, PendingRequests
from tllm.network.manager import RPCManager


def setup_seed(seed: int = 42):
    if BACKEND == BackendEnum.TORCH:
        import torch

        torch.manual_seed(seed)


def init_rpc_manager(client_size: int) -> RPCManager:
    pending_requests = PendingRequests()
    master_handler = MasterHandler(pending_requests)
    rpc_manager = RPCManager(client_size, pending_requests)
    return rpc_manager, master_handler
