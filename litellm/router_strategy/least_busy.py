#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Router calls increment_request_count() before making a request
#   - Router calls decrement_request_count() in a try/finally block after the request completes
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import random
from typing import Optional

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache


class LeastBusyLoggingHandler:
    """
    Tracks in-flight request counts per deployment for least-busy routing.

    Instead of using callbacks (which can miss decrements on streaming disconnects),
    the router calls increment/decrement directly with try/finally guarantees.
    """

    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache

    # TTL of 1800s (30 min) to handle long-running streaming requests
    REQUEST_COUNT_TTL = 1800

    def _get_request_count_cache_key(self, model_group: str, deployment_id: str) -> str:
        """
        Get the cache key for a specific deployment's request count.
        Uses individual keys per deployment for atomic operations.
        """
        return f"deployment:{model_group}:{deployment_id}:request_count"

    def increment_request_count(self, model_group: str, deployment_id: str) -> int:
        """
        Sync: atomically increment the in-flight request count for a deployment.
        Called by the router before making a request.
        Returns the new count.
        """
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = self.router_cache.increment_cache(
            key=cache_key, value=1, ttl=self.REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "least-busy increment: deployment_id=%s, model_group=%s, new_count=%s",
            deployment_id, model_group, new_value,
        )
        return new_value

    def decrement_request_count(self, model_group: str, deployment_id: str) -> int:
        """
        Sync: atomically decrement the in-flight request count for a deployment.
        Called by the router in a finally block after a request completes.
        Ensures count never goes below 0.
        Returns the new count.
        """
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = self.router_cache.increment_cache(
            key=cache_key, value=-1, ttl=self.REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "least-busy decrement: deployment_id=%s, model_group=%s, new_count=%s",
            deployment_id, model_group, new_value,
        )
        if new_value < 0:
            verbose_router_logger.warning(
                "least-busy: negative count for deployment_id=%s, resetting to 0",
                deployment_id,
            )
            self.router_cache.set_cache(key=cache_key, value=0, ttl=self.REQUEST_COUNT_TTL)
            return 0
        return new_value

    async def async_increment_request_count(self, model_group: str, deployment_id: str) -> int:
        """
        Async: atomically increment the in-flight request count for a deployment.
        Called by the router before making a request.
        Returns the new count.
        """
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = await self.router_cache.async_increment_cache(
            key=cache_key, value=1, ttl=self.REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "least-busy async increment: deployment_id=%s, model_group=%s, new_count=%s",
            deployment_id, model_group, new_value,
        )
        return new_value

    async def async_decrement_request_count(self, model_group: str, deployment_id: str) -> int:
        """
        Async: atomically decrement the in-flight request count for a deployment.
        Called by the router in a finally block after a request completes.
        Ensures count never goes below 0.
        Returns the new count.
        """
        if isinstance(deployment_id, int):
            deployment_id = str(deployment_id)
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        new_value = await self.router_cache.async_increment_cache(
            key=cache_key, value=-1, ttl=self.REQUEST_COUNT_TTL
        )
        verbose_router_logger.debug(
            "least-busy async decrement: deployment_id=%s, model_group=%s, new_count=%s",
            deployment_id, model_group, new_value,
        )
        if new_value < 0:
            verbose_router_logger.warning(
                "least-busy: negative count for deployment_id=%s, resetting to 0",
                deployment_id,
            )
            await self.router_cache.async_set_cache(
                key=cache_key, value=0, ttl=self.REQUEST_COUNT_TTL
            )
            return 0
        return new_value

    def _get_request_counts_for_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> dict:
        """
        Sync helper to get request counts for all healthy deployments.
        Returns a dict of {deployment_id: request_count}.

        Uses redis_only=True to bypass in-memory cache and always read from Redis.
        This is critical for distributed deployments where multiple pods need to see
        the global request count, not their local stale view.
        """
        result = {}
        none_count = 0
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            # Use redis_only=True to get global count across all pods
            count = self.router_cache.get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            # Default to 0 if not in cache, ensure non-negative
            result[deployment_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                "least-busy: Redis returned None for all deployments - "
                "Redis may be unavailable. Falling back to random routing."
            )
        return result

    async def _async_get_request_counts_for_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> dict:
        """
        Async helper to get request counts for all healthy deployments.
        Returns a dict of {deployment_id: request_count}.

        Uses redis_only=True to bypass in-memory cache and always read from Redis.
        This is critical for distributed deployments where multiple pods need to see
        the global request count, not their local stale view.
        """
        result = {}
        none_count = 0
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            # Use redis_only=True to get global count across all pods
            count = await self.router_cache.async_get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            # Default to 0 if not in cache, ensure non-negative
            result[deployment_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                "least-busy: Redis returned None for all deployments - "
                "Redis may be unavailable. Falling back to random routing."
            )
        return result

    def _get_available_deployments(
        self,
        healthy_deployments: list,
        all_deployments: dict,
    ):
        """
        Helper to get deployments using least busy strategy.

        When multiple deployments have the same minimum traffic count,
        randomly select among them to ensure fair distribution.
        """
        verbose_router_logger.debug(
            "least-busy: deployment counts=%s", all_deployments
        )

        # First pass: find the minimum traffic count
        min_traffic = float("inf")
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            traffic = all_deployments.get(deployment_id, 0)
            if traffic < min_traffic:
                min_traffic = traffic

        # Second pass: collect all deployments with minimum traffic
        min_deployments = []
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            traffic = all_deployments.get(deployment_id, 0)
            if traffic == min_traffic:
                min_deployments.append(d)

        # Randomly select among deployments with equal minimum traffic
        if min_deployments:
            selected = random.choice(min_deployments)
            verbose_router_logger.debug(
                "least-busy: selected deployment_id=%s with traffic=%s (from %d candidates)",
                selected["model_info"]["id"], min_traffic, len(min_deployments),
            )
            return selected
        else:
            # Fallback: should not happen if healthy_deployments is non-empty
            verbose_router_logger.warning(
                "least-busy: no deployment found, falling back to random choice"
            )
            return random.choice(healthy_deployments)

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ):
        """
        Sync helper to get deployments using least busy strategy
        """
        all_deployments = self._get_request_counts_for_deployments(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )

    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: list
    ):
        """
        Async helper to get deployments using least busy strategy
        """
        all_deployments = await self._async_get_request_counts_for_deployments(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )
