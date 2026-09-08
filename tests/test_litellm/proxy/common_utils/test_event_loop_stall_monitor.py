import asyncio
import time
from unittest.mock import patch

import pytest

from litellm.proxy.common_utils.event_loop_stall_monitor import EventLoopStallMonitor


@pytest.mark.asyncio
async def test_logs_only_delays_above_the_stall_threshold() -> None:
    monitor = EventLoopStallMonitor(interval_seconds=1.0, stall_threshold_seconds=1.0)

    with patch(
        "litellm.proxy.common_utils.event_loop_stall_monitor.verbose_proxy_logger.warning"
    ) as warning:
        monitor.log_recovered_stall(observed_delay_seconds=0.99)
        warning.assert_not_called()

        monitor.log_recovered_stall(observed_delay_seconds=1.25)

    warning.assert_called_once()
    assert "EVENT_LOOP_STALL_RECOVERED" in warning.call_args.args[0]
    assert warning.call_args.args[1] == 1250


@pytest.mark.asyncio
async def test_monitor_detects_an_event_loop_stall() -> None:
    monitor = EventLoopStallMonitor(interval_seconds=0.01, stall_threshold_seconds=0.01)

    with patch(
        "litellm.proxy.common_utils.event_loop_stall_monitor.verbose_proxy_logger.warning"
    ) as warning:
        monitor.start()
        await asyncio.sleep(0.02)
        time.sleep(0.03)
        await asyncio.sleep(0.02)
        await monitor.stop()

    assert warning.call_count >= 1


def test_stop_cancels_the_monitor_task() -> None:
    async def run_test() -> None:
        monitor = EventLoopStallMonitor(interval_seconds=60.0, stall_threshold_seconds=1.0)
        monitor.start()
        assert monitor.is_running is True

        await monitor.stop()

        assert monitor.is_running is False

    asyncio.run(run_test())
