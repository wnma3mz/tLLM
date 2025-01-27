import asyncio
from typing import Any, Dict, Tuple


class StatusTracker:
    def __init__(self, target_count: int):
        self.target_count = target_count
        self.current_count = 0
        self.is_completed = False
        self.future = asyncio.Future()
        self.pp_cost_time = [0 for _ in range(target_count)]

    def update(self, count: int, result: Tuple[int, float]):
        self.current_count = count
        self.pp_cost_time[result[0]] = result[1]
        if self.current_count >= self.target_count:
            self.is_completed = True
            self.future.set_result(self.pp_cost_time)


class PendingRequests:
    def __init__(self):
        self._forward_requests: Dict[str, asyncio.Future] = {}
        self._status_requests: Dict[str, StatusTracker] = {}

    def add_request(self, trace_id: str, pp_size: int) -> Tuple[asyncio.Future, asyncio.Future]:
        forward_future = asyncio.Future()
        self._forward_requests[trace_id] = forward_future
        status_tracker = StatusTracker(pp_size)
        self._status_requests[trace_id] = status_tracker
        return forward_future, status_tracker.future

    def complete_forward_request(self, trace_id: str, result: Any) -> bool:
        if trace_id in self._forward_requests:
            future = self._forward_requests[trace_id]
            if not future.done():
                future.set_result(result)
                del self._forward_requests[trace_id]
                return True
        return False

    def complete_status_request(self, trace_id: str, result: Any) -> bool:
        if trace_id in self._status_requests:
            tracker = self._status_requests[trace_id]
            tracker.update(tracker.current_count + 1, result)
            return tracker.is_completed
        return False

    def fail_forward_request(self, trace_id: str, error: Exception):
        if trace_id in self._forward_requests:
            future = self._forward_requests[trace_id]
            if not future.done():
                future.set_exception(error)
            del self._forward_requests[trace_id]

    def fail_status_request(self, trace_id: str, error: Exception):
        if trace_id in self._status_requests:
            tracker = self._status_requests[trace_id]
            if not tracker.future.done():
                tracker.future.set_exception(error)
            del self._status_requests[trace_id]
