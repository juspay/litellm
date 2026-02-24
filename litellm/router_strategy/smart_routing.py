"""
Smart Routing Strategy

Combines multiple signals to pick the best deployment:
- In-flight request counts (same mechanism as least-busy, called from router try/finally)
- vLLM metrics polling (requests_waiting, requests_running, gpu_cache_usage from /metrics)
- Latency EWMA (exponentially weighted moving average of response latencies)
- Error penalty (recent errors penalize a deployment's score)

Scoring formula:
    score = inflight * W_INFLIGHT
          + ewma_latency * W_LATENCY
          + vllm_queue * W_VLLM_QUEUE
          + gpu_cache * W_GPU_CACHE
          + error_penalty * W_ERROR

Lower score = better deployment.
"""

import asyncio
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.router_strategy.base_routing_strategy import BaseRoutingStrategy


# ── Weight defaults ──────────────────────────────────────────────────────────
DEFAULT_W_INFLIGHT = 1.0
DEFAULT_W_LATENCY = 0.5
DEFAULT_W_VLLM_QUEUE = 2.0
DEFAULT_W_GPU_CACHE = 1.0
DEFAULT_W_ERROR = 5.0

# ── EWMA ─────────────────────────────────────────────────────────────────────
DEFAULT_EWMA_ALPHA = 0.3  # higher = more responsive to recent latency
DEFAULT_INITIAL_LATENCY = 1.0  # seconds, assumed until we have data

# ── Error penalty ────────────────────────────────────────────────────────────
DEFAULT_ERROR_WINDOW = 60.0  # seconds to look back for errors
DEFAULT_ERROR_PENALTY_PER_ERROR = 5.0

# ── vLLM polling ─────────────────────────────────────────────────────────────
DEFAULT_VLLM_POLL_INTERVAL = 5.0  # seconds between polls
DEFAULT_VLLM_POLL_TIMEOUT = 2.0  # seconds per HTTP request

# ── TTL for request counts ───────────────────────────────────────────────────
REQUEST_COUNT_TTL = 1800  # 30 minutes


class SmartRoutingHandler(BaseRoutingStrategy):
    """
    Smart routing handler that uses multiple signals to select the best deployment.
    """

    def __init__(
        self,
        dual_cache: DualCache,
        should_batch_redis_writes: bool = True,
        default_sync_interval: Optional[Union[int, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        vllm_poll_interval: float = DEFAULT_VLLM_POLL_INTERVAL,
    ):
        super().__init__(
            dual_cache=dual_cache,
            should_batch_redis_writes=should_batch_redis_writes,
            default_sync_interval=default_sync_interval,
        )

        # Configurable weights
        w = weights or {}
        self.w_inflight = w.get("inflight", DEFAULT_W_INFLIGHT)
        self.w_latency = w.get("latency", DEFAULT_W_LATENCY)
        self.w_vllm_queue = w.get("vllm_queue", DEFAULT_W_VLLM_QUEUE)
        self.w_gpu_cache = w.get("gpu_cache", DEFAULT_W_GPU_CACHE)
        self.w_error = w.get("error", DEFAULT_W_ERROR)

        # ── In-flight request tracking ───────────────────────────────────
        # Uses the same DualCache-based approach as least_busy for cross-pod consistency.
        # The router calls increment/decrement directly via try/finally.

        # ── Latency EWMA per deployment ──────────────────────────────────
        # Key: deployment_id -> EWMA latency in seconds
        self._ewma_latency: Dict[str, float] = defaultdict(lambda: DEFAULT_INITIAL_LATENCY)

        # ── Error tracking per deployment ────────────────────────────────
        # Key: deployment_id -> list of error timestamps
        self._error_timestamps: Dict[str, List[float]] = defaultdict(list)

        # ── vLLM metrics per deployment ──────────────────────────────────
        # Key: deployment_id -> dict with keys: requests_waiting, requests_running, gpu_cache_usage
        self._vllm_metrics: Dict[str, Dict[str, float]] = {}
        self._vllm_endpoints: Dict[str, str] = {}  # deployment_id -> metrics URL
        self._vllm_poll_interval = vllm_poll_interval
        self._vllm_poll_task: Optional[asyncio.Task] = None

        # HTTP client for vLLM polling
        self._http_client: Optional[httpx.AsyncClient] = None

    # ── Cache key helpers ────────────────────────────────────────────────────

    def _get_request_count_cache_key(self, model_group: str, deployment_id: str) -> str:
        return f"smart_routing:{model_group}:{deployment_id}:request_count"

    # ── Inflight tracking (called from router try/finally) ───────────────────

    def increment_request_count(self, model_group: str, deployment_id: str) -> int:
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = self.dual_cache.increment_cache(
            key=cache_key, value=1, ttl=REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "smart-routing increment: deployment_id=%s, count=%s",
            deployment_id, new_value,
        )
        return new_value

    def decrement_request_count(self, model_group: str, deployment_id: str) -> int:
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = self.dual_cache.increment_cache(
            key=cache_key, value=-1, ttl=REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "smart-routing decrement: deployment_id=%s, count=%s",
            deployment_id, new_value,
        )
        if new_value < 0:
            self.dual_cache.set_cache(key=cache_key, value=0, ttl=REQUEST_COUNT_TTL)
            return 0
        return new_value

    async def async_increment_request_count(self, model_group: str, deployment_id: str) -> int:
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = await self.dual_cache.async_increment_cache(
            key=cache_key, value=1, ttl=REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "smart-routing async increment: deployment_id=%s, count=%s",
            deployment_id, new_value,
        )
        return new_value

    async def async_decrement_request_count(self, model_group: str, deployment_id: str) -> int:
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = await self.dual_cache.async_increment_cache(
            key=cache_key, value=-1, ttl=REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "smart-routing async decrement: deployment_id=%s, count=%s",
            deployment_id, new_value,
        )
        if new_value < 0:
            await self.dual_cache.async_set_cache(
                key=cache_key, value=0, ttl=REQUEST_COUNT_TTL
            )
            return 0
        return new_value

    # ── Latency EWMA ────────────────────────────────────────────────────────

    def record_latency(self, deployment_id: str, latency_seconds: float) -> None:
        """Record a successful response latency for EWMA calculation."""
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        prev = self._ewma_latency[deployment_id]
        self._ewma_latency[deployment_id] = (
            DEFAULT_EWMA_ALPHA * latency_seconds + (1 - DEFAULT_EWMA_ALPHA) * prev
        )

    # ── Error tracking ──────────────────────────────────────────────────────

    def record_error(self, deployment_id: str) -> None:
        """Record an error for the given deployment."""
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        self._error_timestamps[deployment_id].append(time.time())

    def _get_recent_error_count(self, deployment_id: str) -> int:
        """Count errors within the error window."""
        now = time.time()
        cutoff = now - DEFAULT_ERROR_WINDOW
        timestamps = self._error_timestamps.get(deployment_id, [])
        # Prune old entries while counting
        recent = [t for t in timestamps if t >= cutoff]
        self._error_timestamps[deployment_id] = recent
        return len(recent)

    # ── vLLM metrics polling ────────────────────────────────────────────────

    def register_vllm_endpoint(self, deployment_id: str, metrics_url: str) -> None:
        """Register a vLLM deployment's /metrics endpoint for polling."""
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        self._vllm_endpoints[deployment_id] = metrics_url

    def start_vllm_polling(self) -> None:
        """Start the background vLLM metrics polling task."""
        if self._vllm_poll_task is not None or not self._vllm_endpoints:
            return
        try:
            loop = asyncio.get_running_loop()
            self._vllm_poll_task = loop.create_task(self._poll_vllm_metrics_loop())
        except RuntimeError:
            pass

    async def _poll_vllm_metrics_loop(self) -> None:
        """Background loop that polls all registered vLLM endpoints."""
        while True:
            try:
                await self._poll_all_vllm_endpoints()
            except Exception as e:
                verbose_router_logger.debug("vLLM poll error: %s", e)
            await asyncio.sleep(self._vllm_poll_interval)

    async def _poll_all_vllm_endpoints(self) -> None:
        """Poll all registered vLLM /metrics endpoints."""
        if not self._vllm_endpoints:
            return

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=DEFAULT_VLLM_POLL_TIMEOUT)

        tasks = []
        dep_ids = []
        for dep_id, url in self._vllm_endpoints.items():
            dep_ids.append(dep_id)
            tasks.append(self._fetch_vllm_metrics(url))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for dep_id, result in zip(dep_ids, results):
            if isinstance(result, dict):
                self._vllm_metrics[dep_id] = result
            else:
                verbose_router_logger.debug(
                    "vLLM metrics fetch failed for %s: %s", dep_id, result
                )

    async def _fetch_vllm_metrics(self, url: str) -> Dict[str, float]:
        """Fetch and parse Prometheus metrics from a vLLM /metrics endpoint."""
        assert self._http_client is not None
        resp = await self._http_client.get(url)
        resp.raise_for_status()
        return self._parse_prometheus_metrics(resp.text)

    @staticmethod
    def _parse_prometheus_metrics(text: str) -> Dict[str, float]:
        """Extract relevant metrics from Prometheus text format."""
        metrics: Dict[str, float] = {
            "requests_waiting": 0.0,
            "requests_running": 0.0,
            "gpu_cache_usage": 0.0,
        }
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if "vllm:num_requests_waiting" in line:
                try:
                    metrics["requests_waiting"] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif "vllm:num_requests_running" in line:
                try:
                    metrics["requests_running"] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif "vllm:gpu_cache_usage_perc" in line:
                try:
                    metrics["gpu_cache_usage"] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        return metrics

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_deployment(
        self,
        deployment_id: str,
        inflight_count: int,
    ) -> float:
        """
        Compute a composite score for a deployment. Lower is better.
        """
        # Inflight requests
        score = inflight_count * self.w_inflight

        # EWMA latency
        ewma = self._ewma_latency.get(deployment_id, DEFAULT_INITIAL_LATENCY)
        score += ewma * self.w_latency

        # vLLM metrics (if available)
        vllm = self._vllm_metrics.get(deployment_id)
        if vllm is not None:
            score += vllm.get("requests_waiting", 0) * self.w_vllm_queue
            score += vllm.get("gpu_cache_usage", 0) * self.w_gpu_cache

        # Error penalty
        error_count = self._get_recent_error_count(deployment_id)
        score += error_count * DEFAULT_ERROR_PENALTY_PER_ERROR * self.w_error

        return score

    # ── Deployment selection ────────────────────────────────────────────────

    async def _async_get_inflight_counts(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> Dict[str, int]:
        """Get inflight counts for all deployments from cache."""
        result = {}
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            count = await self.dual_cache.async_get_cache(key=cache_key, redis_only=True)
            result[deployment_id] = max(0, int(count)) if count is not None else 0
        return result

    def _get_inflight_counts(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> Dict[str, int]:
        """Get inflight counts for all deployments from cache (sync)."""
        result = {}
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            count = self.dual_cache.get_cache(key=cache_key, redis_only=True)
            result[deployment_id] = max(0, int(count)) if count is not None else 0
        return result

    def _select_deployment(
        self,
        healthy_deployments: list,
        inflight_counts: Dict[str, int],
    ) -> Any:
        """Score all deployments and select the best one."""
        best_score = float("inf")
        best_deployments: list = []

        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            inflight = inflight_counts.get(deployment_id, 0)
            score = self._score_deployment(deployment_id, inflight)

            verbose_router_logger.debug(
                "smart-routing score: deployment_id=%s, inflight=%d, score=%.3f",
                deployment_id, inflight, score,
            )

            if score < best_score:
                best_score = score
                best_deployments = [d]
            elif score == best_score:
                best_deployments.append(d)

        if best_deployments:
            selected = random.choice(best_deployments)
            verbose_router_logger.debug(
                "smart-routing selected: deployment_id=%s (score=%.3f, from %d candidates)",
                selected["model_info"]["id"], best_score, len(best_deployments),
            )
            return selected
        else:
            return random.choice(healthy_deployments)

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        **kwargs: Any,
    ) -> Any:
        """Async: select the best deployment based on composite scoring."""
        inflight_counts = await self._async_get_inflight_counts(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._select_deployment(
            healthy_deployments=healthy_deployments,
            inflight_counts=inflight_counts,
        )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        **kwargs: Any,
    ) -> Any:
        """Sync: select the best deployment based on composite scoring."""
        inflight_counts = self._get_inflight_counts(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._select_deployment(
            healthy_deployments=healthy_deployments,
            inflight_counts=inflight_counts,
        )

    # ── Cleanup ─────────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """Cleanup background tasks and HTTP client."""
        await super().cleanup()
        if self._vllm_poll_task is not None:
            self._vllm_poll_task.cancel()
            try:
                await self._vllm_poll_task
            except asyncio.CancelledError:
                pass
        if self._http_client is not None:
            await self._http_client.aclose()
