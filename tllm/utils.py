from tllm import BACKEND, BackendEnum
from tllm.entrypoints.handler.master_handler import MasterHandler, PendingRequests
from tllm.network.manager import LocalRPCManager, RPCManager


def setup_seed(seed: int = 42):
    if BACKEND == BackendEnum.TORCH:
        import torch

        torch.manual_seed(seed)


async def init_rpc_manager(model_path: str, client_size: int, master_handler_port: int, is_local: bool) -> RPCManager:
    master_handler = None
    if is_local:
        rpc_manager = LocalRPCManager(model_path)
    else:
        pending_requests = PendingRequests()
        master_handler = MasterHandler(pending_requests)
        await master_handler.start(master_handler_port)
        rpc_manager = RPCManager(client_size, pending_requests)
    return rpc_manager, master_handler
