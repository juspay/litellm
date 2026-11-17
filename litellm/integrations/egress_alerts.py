import asyncio
from collections.abc import Mapping
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.alert_system import AlertSystem, alert_judge_active, log_alert_failure


class EgressAlertLogger(CustomLogger):
    def __init__(
        self,
        alert_system: AlertSystem | None = None,
        *,
        concurrency: int = 4,
        max_pending: int = 64,
    ) -> None:
        super().__init__()
        self._system = alert_system if alert_system is not None else AlertSystem.from_env()
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._max_pending = max(1, max_pending)
        self._tasks: frozenset[asyncio.Task[None]] = frozenset()
        self._inflight: frozenset[str] = frozenset()
        self._completed: tuple[str, ...] = ()

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        if alert_judge_active.get() or not self._system.enabled:
            return
        call_id = kwargs.get("litellm_call_id")
        request_id = call_id if isinstance(call_id, str) else ""
        if request_id and (request_id in self._inflight or request_id in self._completed):
            return
        if len(self._tasks) >= self._max_pending:
            log_alert_failure("egress_alert_dropped", call_id)
            return
        if request_id:
            self._inflight = self._inflight.union((request_id,))
        task = asyncio.create_task(self._process(kwargs, response_obj, request_id), name="egress-alert")
        self._tasks = self._tasks.union((task,))
        task.add_done_callback(self._completed_task)

    async def _process(self, kwargs: Mapping[str, object], response_obj: object, request_id: str) -> None:
        try:
            async with self._semaphore:
                await self._system.process_success_event(kwargs, response_obj)
            if request_id:
                self._completed = (*self._completed[-1023:], request_id)
        finally:
            self._inflight = self._inflight.difference((request_id,))

    def _completed_task(self, task: asyncio.Task[None]) -> None:
        self._tasks = self._tasks.difference((task,))

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks))


alert_logger = EgressAlertLogger()
