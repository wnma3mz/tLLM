from typing import Tuple

from tllm import BACKEND, BackendEnum
from tllm.grpc.master_service.master_server import MasterServer
from tllm.grpc.master_service.pending_requests import PendingRequests
from tllm.grpc.master_service.worker_manager import WorkerRPCManager


def setup_seed(seed: int = 42):
    if BACKEND == BackendEnum.TORCH:
        import torch

        torch.manual_seed(seed)


def init_grpc_service(client_size: int) -> Tuple[WorkerRPCManager, MasterServer]:
    pending_requests = PendingRequests()
    master_server = MasterServer(pending_requests)
    worker_rpc_manager = WorkerRPCManager(client_size, pending_requests)
    return worker_rpc_manager, master_server
