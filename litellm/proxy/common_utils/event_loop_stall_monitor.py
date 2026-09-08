import asyncio
import time
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.in_flight_requests_middleware import get_in_flight_requests


class EventLoopStallMonitor:
    def __init__(
        self,
        interval_seconds: float = 1.0,
        stall_threshold_seconds: float = 1.0,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._stall_threshold_seconds = stall_threshold_seconds
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._task = asyncio.create_task(
            self._monitor(), name="litellm-event-loop-stall-monitor"
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def log_recovered_stall(self, observed_delay_seconds: float) -> None:
        if observed_delay_seconds < self._stall_threshold_seconds:
            return
        verbose_proxy_logger.warning(
            "EVENT_LOOP_STALL_RECOVERED observed_delay_ms=%d interval_ms=%d "
            "in_flight_requests=%d active_asyncio_tasks=%d",
            round(observed_delay_seconds * 1000),
            round(self._interval_seconds * 1000),
            get_in_flight_requests(),
            len(asyncio.all_tasks()),
        )

    async def _monitor(self) -> None:
        previous_check = time.perf_counter()
        while True:
            await asyncio.sleep(self._interval_seconds)
            current_check = time.perf_counter()
            self.log_recovered_stall(current_check - previous_check - self._interval_seconds)
            previous_check = current_check
