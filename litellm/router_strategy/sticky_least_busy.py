"""
Sticky-Least-Busy routing strategy.

Routes requests from the same conversation to the same deployment (for KV cache reuse
on vLLM/SGLang nodes), but rebalances to the least-busy deployment when the sticky
target is overloaded.

How this works:
  1. Hash the message prefix (all messages except the last) to compute a sticky key.
  2. Map sticky key to a preferred deployment via consistent hashing.
  3. If preferred deployment's in-flight count < threshold * avg_load, use it (sticky).
  4. If overloaded, route to the deployment with the fewest in-flight requests (rebalance).
  5. Track in-flight requests via Redis (atomic increment/decrement) with dedup
     to avoid the streaming bug where log_pre_api_call fires per SSE chunk.
"""

import hashlib
import json
import random
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class StickyLeastBusyLoggingHandler(CustomLogger):
    """
    Routing handler that combines conversation stickiness with load-aware rebalancing.
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(
        self,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        virtual_nodes: int = 150,
        cache_ttl: int = 600,
    ):
        """
        Args:
            router_cache: DualCache instance for Redis + in-memory caching.
            imbalance_threshold: If sticky node load > threshold * avg_load, rebalance.
            virtual_nodes: Number of virtual nodes per deployment on the consistent hash ring.
            cache_ttl: TTL in seconds for request count cache keys.
        """
        self.router_cache = router_cache
        self.imbalance_threshold = imbalance_threshold
        self.virtual_nodes = virtual_nodes
        self.cache_ttl = cache_ttl

        # Streaming dedup: track which litellm_call_ids we've already incremented.
        # log_pre_api_call fires for every SSE chunk in streaming - only increment once.
        self._seen_call_ids: Dict[str, bool] = {}
        self._seen_call_ids_max_size: int = 10000

        # Consistent hash ring (rebuilt when deployments change)
        self._hash_ring: List[Tuple[int, str]] = []
        self._ring_deployment_ids: frozenset = frozenset()

    # =========================================================================
    # Prefix Hashing
    # =========================================================================

    @staticmethod
    def compute_sticky_key(
        messages: Optional[List[Dict[str, str]]],
    ) -> Optional[str]:
        """
        Compute a deterministic hash from the conversation prefix.

        - None/empty messages -> None (no stickiness, degrades to least-busy).
        - Single message -> hash that message.
        - Multiple messages -> hash all except the last (the conversation context).
        - Long conversations (>20 prefix messages) -> first 10 + last 10.

        SHA-256 of canonical JSON ensures cross-pod determinism.
        """
        if not messages:
            return None

        if len(messages) == 1:
            prefix = messages
        else:
            prefix = messages[:-1]

        # Bound hashing cost for very long conversations
        if len(prefix) > 20:
            prefix = prefix[:10] + prefix[-10:]

        try:
            canonical = json.dumps(
                prefix, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            canonical = str(prefix)

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # =========================================================================
    # Consistent Hashing
    # =========================================================================

    def _build_hash_ring(self, deployment_ids: List[str]) -> None:
        """
        Build a consistent hash ring from deployment IDs using virtual nodes.
        Only rebuilds if the set of IDs has changed.
        """
        new_ids = frozenset(deployment_ids)
        if new_ids == self._ring_deployment_ids:
            return

        ring: List[Tuple[int, str]] = []
        for dep_id in deployment_ids:
            for i in range(self.virtual_nodes):
                key = f"{dep_id}:{i}"
                h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
                ring.append((h, dep_id))

        ring.sort(key=lambda x: x[0])
        self._hash_ring = ring
        self._ring_deployment_ids = new_ids

    def _get_deployment_for_key(self, sticky_key: str) -> Optional[str]:
        """Map a sticky key to a deployment ID via the consistent hash ring."""
        if not self._hash_ring:
            return None

        h = int(hashlib.md5(sticky_key.encode("utf-8")).hexdigest(), 16)
        idx = bisect_right(self._hash_ring, (h,))
        if idx >= len(self._hash_ring):
            idx = 0

        return self._hash_ring[idx][1]

    # =========================================================================
    # Request Count Cache Keys
    # =========================================================================

    def _get_request_count_cache_key(
        self, model_group: str, deployment_id: str
    ) -> str:
        return f"sticky_lb:{model_group}:{deployment_id}:request_count"

    # =========================================================================
    # Streaming Dedup
    # =========================================================================

    def _should_increment(self, litellm_call_id: str) -> bool:
        """
        Returns True only for the FIRST call with this litellm_call_id.
        Subsequent calls (SSE streaming chunks) return False.
        """
        if litellm_call_id in self._seen_call_ids:
            return False

        if len(self._seen_call_ids) >= self._seen_call_ids_max_size:
            keys_to_remove = list(self._seen_call_ids.keys())[
                : self._seen_call_ids_max_size // 10
            ]
            for key in keys_to_remove:
                self._seen_call_ids.pop(key, None)

        self._seen_call_ids[litellm_call_id] = True
        return True

    def _cleanup_call_id(self, litellm_call_id: str) -> None:
        self._seen_call_ids.pop(litellm_call_id, None)

    # =========================================================================
    # CustomLogger Callbacks - Request Tracking
    # =========================================================================

    def log_pre_api_call(self, model, messages, kwargs):
        """Increment in-flight count. Deduped by litellm_call_id for streaming."""
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return

            model_group = litellm_params["metadata"].get("model_group")
            dep_id = litellm_params.get("model_info", {}).get("id")
            if model_group is None or dep_id is None:
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id and not self._should_increment(litellm_call_id):
                return

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            self.router_cache.increment_cache(
                key=cache_key, value=1, ttl=self.cache_ttl
            )
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusy log_pre_api_call error: {e}"
            )

    def _decrement_request_count(self, kwargs) -> None:
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = litellm_params.get("model_info", {}).get("id")
            if model_group is None or dep_id is None:
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = self.router_cache.increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            if new_value < 0:
                self.router_cache.set_cache(
                    key=cache_key, value=0, ttl=self.cache_ttl
                )

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id:
                self._cleanup_call_id(litellm_call_id)
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusy decrement error: {e}"
            )

    async def _async_decrement_request_count(self, kwargs) -> None:
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = litellm_params.get("model_info", {}).get("id")
            if model_group is None or dep_id is None:
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = await self.router_cache.async_increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            if new_value < 0:
                await self.router_cache.async_set_cache(
                    key=cache_key, value=0, ttl=self.cache_ttl
                )

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id:
                self._cleanup_call_id(litellm_call_id)
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusy async decrement error: {e}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs)
        if self.test_flag:
            self.logged_success += 1

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs)
        if self.test_flag:
            self.logged_failure += 1

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        await self._async_decrement_request_count(kwargs)
        if self.test_flag:
            self.logged_success += 1

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        await self._async_decrement_request_count(kwargs)
        if self.test_flag:
            self.logged_failure += 1

    # =========================================================================
    # Load Querying
    # =========================================================================

    def _get_request_counts(
        self, model_group: str, healthy_deployments: list
    ) -> Dict[str, int]:
        """Sync: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            count = self.router_cache.get_cache(key=cache_key, redis_only=True)
            result[dep_id] = max(0, int(count)) if count is not None else 0
        return result

    async def _async_get_request_counts(
        self, model_group: str, healthy_deployments: list
    ) -> Dict[str, int]:
        """Async: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            count = await self.router_cache.async_get_cache(
                key=cache_key, redis_only=True
            )
            result[dep_id] = max(0, int(count)) if count is not None else 0
        return result

    # =========================================================================
    # Deployment Selection Core
    # =========================================================================

    def _select_deployment(
        self,
        healthy_deployments: list,
        request_counts: Dict[str, int],
        sticky_key: Optional[str],
    ) -> dict:
        """
        Core selection logic:
        1. Build/update consistent hash ring from healthy deployment IDs.
        2. If sticky_key available, find preferred deployment via consistent hashing.
        3. Check if preferred deployment is within load threshold.
        4. If overloaded or no sticky key, fall back to least-busy.
        """
        dep_id_to_deployment: Dict[str, dict] = {}
        dep_ids: List[str] = []
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            dep_ids.append(dep_id)
            dep_id_to_deployment[dep_id] = d

        self._build_hash_ring(dep_ids)

        total_load = sum(request_counts.get(did, 0) for did in dep_ids)
        avg_load = total_load / len(dep_ids) if dep_ids else 0

        # Try sticky routing
        if sticky_key:
            preferred_id = self._get_deployment_for_key(sticky_key)
            if preferred_id and preferred_id in dep_id_to_deployment:
                preferred_load = request_counts.get(preferred_id, 0)
                effective_avg = max(avg_load, 1.0)
                if preferred_load < self.imbalance_threshold * effective_avg:
                    verbose_router_logger.debug(
                        f"StickyLeastBusy: sticky routing to {preferred_id} "
                        f"(load={preferred_load}, avg={avg_load:.1f}, "
                        f"threshold={self.imbalance_threshold})"
                    )
                    return dep_id_to_deployment[preferred_id]
                else:
                    verbose_router_logger.info(
                        f"StickyLeastBusy: overriding stickiness for {preferred_id} "
                        f"(load={preferred_load} >= "
                        f"{self.imbalance_threshold} * {avg_load:.1f}), "
                        f"falling back to least-busy"
                    )

        # Least-busy fallback with random tie-breaking
        min_load = float("inf")
        for did in dep_ids:
            load = request_counts.get(did, 0)
            if load < min_load:
                min_load = load

        min_deployments = [
            dep_id_to_deployment[did]
            for did in dep_ids
            if request_counts.get(did, 0) == min_load
        ]

        selected = (
            random.choice(min_deployments)
            if min_deployments
            else random.choice(healthy_deployments)
        )
        verbose_router_logger.debug(
            f"StickyLeastBusy: least-busy routing to "
            f"{selected['model_info']['id']} "
            f"(load={min_load}, from {len(min_deployments)} candidates)"
        )
        return selected

    # =========================================================================
    # Public API - Called by Router
    # =========================================================================

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        request_counts = self._get_request_counts(model_group, healthy_deployments)
        sticky_key = self.compute_sticky_key(messages)
        return self._select_deployment(
            healthy_deployments, request_counts, sticky_key
        )

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        request_counts = await self._async_get_request_counts(
            model_group, healthy_deployments
        )
        sticky_key = self.compute_sticky_key(messages)
        return self._select_deployment(
            healthy_deployments, request_counts, sticky_key
        )
