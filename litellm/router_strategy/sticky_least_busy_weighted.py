"""
Weighted Sticky-Least-Busy routing strategy.

Same conversation-stickiness + load-aware rebalancing as `sticky-least-busy`, but each
deployment carries a relative capacity weight (`model_info.sticky_weight`, e.g. 1 for
H200, 2 for B300). This is a SEPARATE strategy (`sticky-least-busy-weighted`) shipped
alongside the unweighted one so it can be toggled/rolled back from the UI.

How this works:
  1. Hash the conversation identity (first user message + user ID) to compute
     a sticky key that is constant across all turns.
  2. Map sticky key to a preferred deployment via a WEIGHTED consistent hash ring —
     a deployment gets `virtual_nodes * sticky_weight` points, so higher-capacity
     nodes own a proportionally larger share of conversations.
  3. Compute a reference load using the avg+min blend: (avg_load + min_load) / 2,
     over CAPACITY-NORMALIZED load (in-flight / weight).
  4. If preferred deployment's normalized load < threshold * reference_load, use it (sticky).
  5. If overloaded, route to the deployment with the lowest normalized load (rebalance).
  6. Track in-flight requests via Redis (atomic increment/decrement) with dedup
     to avoid the streaming bug where log_pre_api_call fires per SSE chunk.

Independent from the base handler: own singleton, own Prometheus metric names
(litellm_sticky_weighted_routing_*), and own Redis key prefix (sticky_lb_weighted:).
"""

import hashlib
import random
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class StickyLeastBusyWeightedLoggingHandler(CustomLogger):
    """
    Routing handler that combines conversation stickiness with load-aware rebalancing.

    Uses a class-level singleton to survive Router re-creation. The LiteLLM proxy
    may create new Router instances per-request or on config syncs. Without a
    singleton, each new instance gets a fresh _seen_call_ids dict, breaking
    streaming dedup and causing in-flight counts to grow monotonically.
    """

    _instance: Optional["StickyLeastBusyWeightedLoggingHandler"] = None

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __new__(
        cls,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        virtual_nodes: int = 150,
        cache_ttl: int = 600,
    ):
        """
        Singleton: return existing instance if one exists.
        Only update router_cache (which may change across Router instances).
        """
        if cls._instance is not None:
            # Update router_cache to the latest Router's cache (may have new Redis connection)
            cls._instance.router_cache = router_cache
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted REUSE] Reusing existing handler "
                f"(seen_call_ids={len(cls._instance._seen_call_ids)}, "
                f"rings={len(cls._instance._rings)})"
            )
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

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
            imbalance_threshold: If sticky node load > threshold * reference_load, rebalance.
                reference_load = (avg_load + min_load) / 2 to catch skewed distributions.
            virtual_nodes: Number of virtual nodes per deployment on the consistent hash ring.
            cache_ttl: TTL in seconds for request count cache keys.
        """
        # Skip re-initialization if already initialized (singleton reuse)
        if hasattr(self, "_initialized") and self._initialized:
            # Always update router_cache (may point to a new Router's cache)
            self.router_cache = router_cache
            return

        self._initialized = True
        self.router_cache = router_cache
        self.imbalance_threshold = imbalance_threshold
        self.virtual_nodes = virtual_nodes
        self.cache_ttl = cache_ttl

        # Streaming dedup: track which litellm_call_ids we've already incremented.
        # log_pre_api_call fires for every SSE chunk in streaming - only increment once.
        self._seen_call_ids: Dict[str, bool] = {}
        self._seen_call_ids_max_size: int = 10000
        self._completed_call_ids: Dict[str, bool] = {}
        self._completed_call_ids_max_size: int = 10000
        self._decremented_call_ids: Dict[str, bool] = {}
        self._decremented_call_ids_max_size: int = 10000
        self._selected_deployments_by_call_id: Dict[str, Tuple[str, str, str]] = {}
        self._selected_deployments_max_size: int = 10000

        # Per-model-group consistent hash rings.
        # Each model group (e.g., "llama-70b", "kimi-k2-5-dev") may have different
        # deployments, so each needs its own ring. Keyed by model_group name.
        # Value: (signature, sorted_ring_list) where signature is a frozenset of
        # (deployment_id, weight) pairs — so a weight change (not just an id-set
        # change) invalidates the cached ring and triggers a rebuild.
        self._rings: Dict[str, Tuple[frozenset, List[Tuple[int, str]]]] = {}

        # Prometheus metrics (lazy init — no-op if prometheus_client not installed)
        try:
            from prometheus_client import Counter, Gauge

            self._routing_decisions = Counter(
                "litellm_sticky_weighted_routing_decisions_total",
                "Routing decisions made by sticky-least-busy strategy",
                ["model_group", "deployment_id", "decision", "strategy"],
            )
            self._routing_in_flight = Gauge(
                "litellm_sticky_weighted_routing_in_flight",
                "In-flight requests per deployment tracked by sticky routing",
                ["model_group", "deployment_id"],
            )
            self._routing_fallback = Counter(
                "litellm_sticky_weighted_routing_fallback_total",
                "Fallback events in sticky routing",
                ["model_group", "reason", "strategy"],
            )
            self._routing_redis_count = Gauge(
                "litellm_sticky_weighted_routing_redis_count",
                "Redis in-flight count per deployment as seen by routing at decision time",
                ["model_group", "deployment_id"],
            )
            self._routing_redis_count_by_load_key = Gauge(
                "litellm_sticky_weighted_routing_redis_count_by_load_key",
                "Redis in-flight count per backend load key as seen by routing at decision time",
                ["load_key", "load_key_source"],
            )
            self._healthy_deployments_count = Gauge(
                "litellm_sticky_weighted_healthy_deployments_count",
                "Healthy deployments available to the weighted sticky router",
                ["model_group"],
            )
            self._deployment_healthy = Gauge(
                "litellm_sticky_weighted_deployment_healthy",
                "Whether a deployment is available to the weighted sticky router",
                ["model_group", "deployment_id"],
            )
            self._deployment_weight = Gauge(
                "litellm_sticky_weighted_deployment_weight",
                "Configured capacity weight used by the weighted sticky router",
                ["model_group", "deployment_id"],
            )
            self._normalized_load = Gauge(
                "litellm_sticky_weighted_normalized_load",
                "Capacity-normalized load used by the weighted sticky router",
                ["model_group", "deployment_id"],
            )
            self._reference_load = Gauge(
                "litellm_sticky_weighted_reference_load",
                "Reference load used for weighted sticky overload decisions",
                ["model_group"],
            )
            self._threshold_load = Gauge(
                "litellm_sticky_weighted_threshold_load",
                "Load threshold above which a sticky deployment is overridden",
                ["model_group"],
            )
            self._counter_resets = Counter(
                "litellm_sticky_weighted_counter_resets_total",
                "Negative in-flight counters reset to zero",
                ["model_group", "deployment_id"],
            )
            self._counter_events = Counter(
                "litellm_sticky_weighted_counter_events_total",
                "In-flight counter bookkeeping events",
                ["action", "result", "load_key_source"],
            )
        except ValueError:
            # Already registered by another handler instance — reuse from registry
            from prometheus_client import REGISTRY

            self._routing_decisions = REGISTRY._names_to_collectors.get(
                "litellm_sticky_weighted_routing_decisions_total"
            )
            self._routing_in_flight = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_routing_in_flight")
            self._routing_fallback = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_routing_fallback_total")
            self._routing_redis_count = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_routing_redis_count")
            self._routing_redis_count_by_load_key = REGISTRY._names_to_collectors.get(
                "litellm_sticky_weighted_routing_redis_count_by_load_key"
            )
            self._healthy_deployments_count = REGISTRY._names_to_collectors.get(
                "litellm_sticky_weighted_healthy_deployments_count"
            )
            self._deployment_healthy = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_deployment_healthy")
            self._deployment_weight = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_deployment_weight")
            self._normalized_load = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_normalized_load")
            self._reference_load = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_reference_load")
            self._threshold_load = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_threshold_load")
            self._counter_resets = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_counter_resets_total")
            self._counter_events = REGISTRY._names_to_collectors.get("litellm_sticky_weighted_counter_events_total")
            # Guard against partial registration — if any lookup returned None,
            # fall back to NoOpMetric to avoid AttributeError on .labels().inc()
            metrics = (
                self._routing_decisions,
                self._routing_in_flight,
                self._routing_fallback,
                self._routing_redis_count,
                self._routing_redis_count_by_load_key,
                self._healthy_deployments_count,
                self._deployment_healthy,
                self._deployment_weight,
                self._normalized_load,
                self._reference_load,
                self._threshold_load,
                self._counter_resets,
                self._counter_events,
            )
            if not all(metrics):
                from litellm.types.integrations.prometheus import NoOpMetric

                self._routing_decisions = self._routing_decisions or NoOpMetric()
                self._routing_in_flight = self._routing_in_flight or NoOpMetric()
                self._routing_fallback = self._routing_fallback or NoOpMetric()
                self._routing_redis_count = self._routing_redis_count or NoOpMetric()
                self._routing_redis_count_by_load_key = self._routing_redis_count_by_load_key or NoOpMetric()
                self._healthy_deployments_count = self._healthy_deployments_count or NoOpMetric()
                self._deployment_healthy = self._deployment_healthy or NoOpMetric()
                self._deployment_weight = self._deployment_weight or NoOpMetric()
                self._normalized_load = self._normalized_load or NoOpMetric()
                self._reference_load = self._reference_load or NoOpMetric()
                self._threshold_load = self._threshold_load or NoOpMetric()
                self._counter_resets = self._counter_resets or NoOpMetric()
                self._counter_events = self._counter_events or NoOpMetric()
        except Exception:
            from litellm.types.integrations.prometheus import NoOpMetric

            self._routing_decisions = NoOpMetric()
            self._routing_in_flight = NoOpMetric()
            self._routing_fallback = NoOpMetric()
            self._routing_redis_count = NoOpMetric()
            self._routing_redis_count_by_load_key = NoOpMetric()
            self._healthy_deployments_count = NoOpMetric()
            self._deployment_healthy = NoOpMetric()
            self._deployment_weight = NoOpMetric()
            self._normalized_load = NoOpMetric()
            self._reference_load = NoOpMetric()
            self._threshold_load = NoOpMetric()
            self._counter_resets = NoOpMetric()
            self._counter_events = NoOpMetric()

        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted INIT] Initialized with "
            f"imbalance_threshold={imbalance_threshold}, "
            f"virtual_nodes={virtual_nodes}, "
            f"cache_ttl={cache_ttl}s"
        )

    # =========================================================================
    # Prefix Hashing
    # =========================================================================

    @staticmethod
    def compute_sticky_key(
        messages: Optional[List[Dict[str, str]]],
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Compute a deterministic hash that identifies the conversation per user.

        The key must be STABLE across all turns of the same conversation so that
        consecutive messages route to the same node (KV cache reuse). We achieve
        this by hashing the conversation's "identity" — the first user message
        plus a user identifier — which never changes as the conversation grows.

        Algorithm:
        - None/empty messages -> None (no stickiness, degrades to least-busy).
        - Extract the first user message content (O(1) scan, stops at first user msg).
        - Combine with user_id (API key or user ID) for per-user differentiation.
        - Hash with SHA-256.

        This ensures:
        - Same conversation always produces the same hash on every turn.
        - Different conversations (different first user question) get different hashes.
        - The hash is deterministic across pods (SHA-256).
        - Different users with the same system prompt AND same first question
          get different hashes (per-user stickiness, no hotspot).
        """
        if not messages:
            verbose_router_logger.debug("[StickyLeastBusyWeighted STICKY-KEY] No messages provided, sticky_key=None")
            return None

        # Extract the first user message content.
        # O(1) scan — stops at first user message, doesn't touch the rest.
        first_user_content: Optional[str] = None
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")
                # Handle multimodal content (list of parts) — extract text parts
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    first_user_content = " ".join(text_parts) if text_parts else ""
                else:
                    first_user_content = str(content) if content is not None else ""
                break
            elif role in ("system", "developer"):
                continue
            else:
                break

        if first_user_content is None:
            verbose_router_logger.debug("[StickyLeastBusyWeighted STICKY-KEY] No user message found, sticky_key=None")
            return None

        # Combine first user message + user identifier for per-user stickiness.
        # If user_id is not available, fall back to message-only hashing.
        hash_input = first_user_content
        if user_id:
            hash_input = f"{user_id}:{first_user_content}"

        sticky_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        verbose_router_logger.debug(
            f"[StickyLeastBusyWeighted STICKY-KEY] "
            f"total_messages={len(messages)}, "
            f"has_user_id={user_id is not None}, "
            f"sticky_key={sticky_key[:16]}..."
        )
        return sticky_key

    # =========================================================================
    # Consistent Hashing
    # =========================================================================

    def _vnodes_for_weight(self, weight: float) -> int:
        """
        Number of virtual nodes a deployment gets for a given capacity weight.

        Computed in INTEGER space (scale weight to per-mille, multiply, then
        integer-divide) rather than as a float `round(virtual_nodes * weight)`.
        A float value landing exactly on a .5 rounding boundary could round
        differently across pods/architectures, producing divergent rings and
        breaking cross-pod stickiness. Integer math is bit-identical everywhere.
        Always at least 1 vnode so a positive weight is never dropped from the ring.
        """
        return max(1, (self.virtual_nodes * round(weight * 1000)) // 1000)

    def _build_hash_ring(self, model_group: str, weights: Union[Dict[str, float], List[str]]) -> None:
        """
        Build a weighted consistent hash ring from deployment weights.

        Each deployment gets `virtual_nodes * weight` points on the ring, so a
        deployment with a higher capacity weight owns a proportionally larger
        share of the key space (more conversations stick to it).

        `weights` is normally a {deployment_id: weight} dict. A plain list of
        deployment IDs is also accepted as shorthand for equal weight (1.0).

        Rings are cached per model_group and keyed on the full (id, weight) set,
        so the ring rebuilds when a deployment is added/removed OR when any
        deployment's weight changes.
        """
        if not isinstance(weights, dict):
            weights = {dep_id: 1.0 for dep_id in weights}
        signature = frozenset(weights.items())
        cached = self._rings.get(model_group)
        if cached and cached[0] == signature:
            return

        prev_count = len(cached[0]) if cached else 0
        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted RING-BUILD] Rebuilding hash ring for "
            f"model_group={model_group}: "
            f"prev_deployments={prev_count}, "
            f"new_deployments={len(weights)}, "
            f"weights={dict(weights)}"
        )

        ring: List[Tuple[int, str]] = []
        for dep_id, weight in weights.items():
            n = self._vnodes_for_weight(weight)
            for i in range(n):
                # Key scheme unchanged from the unweighted ring: raising a
                # deployment's weight only *appends* new points (i grows), it
                # never moves existing ones — so re-weighting is cache-cheap.
                key = f"{dep_id}:{i}"
                h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
                ring.append((h, dep_id))

        ring.sort(key=lambda x: x[0])
        self._rings[model_group] = (signature, ring)

        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted RING-BUILD] Ring for {model_group} built with "
            f"{len(ring)} virtual nodes "
            f"(base {self.virtual_nodes} per deployment, scaled by weight)"
        )

    def _get_deployment_for_key(self, model_group: str, sticky_key: str) -> Optional[str]:
        """Map a sticky key to a deployment ID via the consistent hash ring."""
        cached = self._rings.get(model_group)
        if not cached or not cached[1]:
            verbose_router_logger.debug(
                f"[StickyLeastBusyWeighted RING-LOOKUP] Hash ring for " f"{model_group} is empty, returning None"
            )
            return None

        ring = cached[1]
        h = int(hashlib.md5(sticky_key.encode("utf-8")).hexdigest(), 16)
        idx = bisect_right(ring, (h,))
        if idx >= len(ring):
            idx = 0

        result = ring[idx][1]
        verbose_router_logger.debug(
            f"[StickyLeastBusyWeighted RING-LOOKUP] "
            f"model_group={model_group}, "
            f"sticky_key={sticky_key[:16]}... -> deployment_id={result}"
        )
        return result

    # =========================================================================
    # Request Count Cache Keys
    # =========================================================================

    def _get_request_count_cache_key(
        self, _model_group: str, deployment_id: str, load_key: Optional[str] = None
    ) -> str:
        if load_key is None:
            load_key = f"deployment:{deployment_id}"
        return f"sticky_lb_weighted:load:{load_key}:request_count"

    @staticmethod
    def _normalize_load_key_part(value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text.rstrip("/")

    @staticmethod
    def _explicit_load_key_from_model_info(model_info: Optional[dict]) -> Optional[str]:
        if not isinstance(model_info, dict):
            return None
        for key in ("sticky_load_key", "capacity_pool_id", "load_key"):
            value = StickyLeastBusyWeightedLoggingHandler._normalize_load_key_part(model_info.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _resolve_load_key(
        cls,
        model_info: Optional[dict],
        deployed_model: object,
        api_base: object,
        deployment_id: object,
    ) -> Tuple[Optional[str], str]:
        explicit_load_key = cls._explicit_load_key_from_model_info(model_info)
        if explicit_load_key is not None:
            return f"explicit:{explicit_load_key}", "explicit"

        normalized_model = cls._normalize_load_key_part(deployed_model)
        normalized_api_base = cls._normalize_load_key_part(api_base)
        if normalized_model is not None and normalized_api_base is not None:
            digest = hashlib.sha256(f"{normalized_model}|{normalized_api_base}".encode("utf-8")).hexdigest()
            return f"backend:{digest}", "backend_model_api_base"

        normalized_deployment_id = cls._normalize_load_key_part(deployment_id)
        if normalized_deployment_id is not None:
            return f"deployment:{normalized_deployment_id}", "deployment_id"

        return None, "missing"

    @classmethod
    def _get_load_key_for_deployment(cls, deployment: dict) -> Tuple[Optional[str], str]:
        model_info = deployment.get("model_info") if isinstance(deployment, dict) else None
        litellm_params = deployment.get("litellm_params") if isinstance(deployment, dict) else None
        deployed_model = litellm_params.get("model") if isinstance(litellm_params, dict) else None
        api_base = litellm_params.get("api_base") if isinstance(litellm_params, dict) else None
        deployment_id = model_info.get("id") if isinstance(model_info, dict) else None
        return cls._resolve_load_key(model_info, deployed_model, api_base, deployment_id)

    @staticmethod
    def _format_load_key(load_key: Optional[str]) -> str:
        if load_key is None:
            return "None"
        return load_key if len(load_key) <= 40 else f"{load_key[:40]}..."

    @staticmethod
    def _format_call_id(litellm_call_id: object) -> str:
        if litellm_call_id is None:
            return "None"
        call_id = str(litellm_call_id)
        if not call_id:
            return "None"
        return f"{call_id[:16]}..."

    @staticmethod
    def _previous_count_from_delta(new_value: object, delta: int) -> str:
        if isinstance(new_value, (int, float)):
            return str(new_value - delta)
        return "unknown"

    def _is_redis_configured(self) -> bool:
        return self.router_cache.redis_cache is not None

    @staticmethod
    def _extract_litellm_call_id_from_kwargs(kwargs) -> Optional[str]:
        if kwargs is None:
            return None

        call_id = kwargs.get("litellm_call_id")
        if call_id:
            return str(call_id)

        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            call_id = litellm_params.get("litellm_call_id")
            if call_id:
                return str(call_id)

            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                call_id = metadata.get("litellm_call_id")
                if call_id:
                    return str(call_id)

        metadata = kwargs.get("metadata") or kwargs.get("litellm_metadata")
        if isinstance(metadata, dict):
            call_id = metadata.get("litellm_call_id")
            if call_id:
                return str(call_id)

        return None

    def _get_counter_tracking_info(
        self, kwargs
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str, str, bool]:
        litellm_call_id = self._extract_litellm_call_id_from_kwargs(kwargs)
        litellm_params = kwargs.get("litellm_params") if kwargs is not None else None
        source = "litellm_params"

        metadata = None
        if isinstance(litellm_params, dict):
            litellm_metadata = litellm_params.get("metadata")
            if isinstance(litellm_metadata, dict):
                metadata = litellm_metadata

        if metadata is None and kwargs is not None:
            top_level_metadata = kwargs.get("metadata")
            if isinstance(top_level_metadata, dict):
                metadata = top_level_metadata
                source = "metadata"
            else:
                litellm_metadata = kwargs.get("litellm_metadata")
                if isinstance(litellm_metadata, dict):
                    metadata = litellm_metadata
                    source = "litellm_metadata"

        model_group = metadata.get("model_group") if isinstance(metadata, dict) else None

        model_info = None
        if isinstance(litellm_params, dict):
            litellm_model_info = litellm_params.get("model_info")
            if isinstance(litellm_model_info, dict):
                model_info = litellm_model_info

        if model_info is None and kwargs is not None:
            top_level_model_info = kwargs.get("model_info")
            if isinstance(top_level_model_info, dict):
                model_info = top_level_model_info
                if source == "litellm_params":
                    source = "model_info"

        if model_info is None and isinstance(metadata, dict):
            metadata_model_info = metadata.get("model_info")
            if isinstance(metadata_model_info, dict):
                model_info = metadata_model_info
                source = f"{source}.model_info"

        dep_id = model_info.get("id") if isinstance(model_info, dict) else None
        deployed_model = None
        api_base = None
        if isinstance(metadata, dict):
            deployed_model = metadata.get("deployment")
            api_base = metadata.get("api_base")
        if deployed_model is None and isinstance(litellm_params, dict):
            deployed_model = litellm_params.get("model")
        if api_base is None and isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")
        if kwargs is not None:
            if deployed_model is None:
                deployed_model = kwargs.get("model")
            if api_base is None:
                api_base = kwargs.get("api_base")

        load_key, load_key_source = self._resolve_load_key(model_info, deployed_model, api_base, dep_id)
        route_map_hit = False
        if litellm_call_id and (
            model_group is None or dep_id is None or load_key is None or load_key_source == "deployment_id"
        ):
            selected_deployment = self._selected_deployments_by_call_id.get(litellm_call_id)
            if selected_deployment is not None:
                if model_group is None:
                    model_group = selected_deployment[0]
                if dep_id is None:
                    dep_id = selected_deployment[1]
                if (load_key is None or load_key_source == "deployment_id") and len(selected_deployment) > 2:
                    load_key = selected_deployment[2]
                    load_key_source = "route_map"
                route_map_hit = True
                source = f"{source}+route_map"

        if model_group is not None:
            model_group = str(model_group)
        if dep_id is not None:
            dep_id = str(dep_id)

        return model_group, dep_id, load_key, litellm_call_id, source, load_key_source, route_map_hit

    def _get_counter_skip_context(self, kwargs) -> str:
        if kwargs is None:
            return "kwargs_none"

        litellm_params = kwargs.get("litellm_params")
        litellm_metadata = litellm_params.get("metadata") if isinstance(litellm_params, dict) else None
        top_level_metadata = kwargs.get("metadata")
        litellm_top_level_metadata = kwargs.get("litellm_metadata")
        top_level_model_info = kwargs.get("model_info")
        has_any_metadata = any(
            isinstance(candidate, dict)
            for candidate in (
                litellm_metadata,
                top_level_metadata,
                litellm_top_level_metadata,
            )
        )

        if not isinstance(litellm_params, dict) and not has_any_metadata:
            reason = "missing_litellm_params_or_metadata"
        else:
            reason = "missing_model_group_or_deployment_id"

        return (
            f"reason={reason}, has_litellm_params={isinstance(litellm_params, dict)}, "
            f"has_litellm_metadata={isinstance(litellm_metadata, dict)}, "
            f"has_top_level_metadata={isinstance(top_level_metadata, dict)}, "
            f"has_litellm_top_level_metadata={isinstance(litellm_top_level_metadata, dict)}, "
            f"has_top_level_model_info={isinstance(top_level_model_info, dict)}"
        )

    def _remember_selected_deployment_for_kwargs(
        self,
        request_kwargs: Optional[Dict],
        model_group: str,
        selected_deployment: dict,
    ) -> None:
        litellm_call_id = self._extract_litellm_call_id_from_kwargs(request_kwargs)
        if not litellm_call_id:
            return

        dep_id = (selected_deployment.get("model_info") or {}).get("id")
        if dep_id is None:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted ROUTE-MAP-SKIP] reason=missing_deployment_id "
                f"model_group={model_group}, call_id={self._format_call_id(litellm_call_id)}"
            )
            return

        if len(self._selected_deployments_by_call_id) >= self._selected_deployments_max_size:
            evict_count = max(1, self._selected_deployments_max_size // 10)
            keys_to_remove = list(self._selected_deployments_by_call_id.keys())[:evict_count]
            for key in keys_to_remove:
                self._selected_deployments_by_call_id.pop(key, None)

        load_key, load_key_source = self._get_load_key_for_deployment(selected_deployment)
        if load_key is None:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted ROUTE-MAP-SKIP] reason=missing_load_key "
                f"model_group={model_group}, deployment_id={dep_id}, "
                f"call_id={self._format_call_id(litellm_call_id)}"
            )
            return

        self._selected_deployments_by_call_id[litellm_call_id] = (str(model_group), str(dep_id), load_key)
        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted ROUTE-MAP] action=remember "
            f"model_group={model_group}, deployment_id={dep_id}, "
            f"load_key={self._format_load_key(load_key)}, "
            f"load_key_source={load_key_source}, "
            f"call_id={self._format_call_id(litellm_call_id)}, "
            f"tracked_call_ids={len(self._selected_deployments_by_call_id)}"
        )

    def _remember_completed_call_id(self, litellm_call_id: str) -> None:
        if len(self._completed_call_ids) >= self._completed_call_ids_max_size:
            evict_count = max(1, self._completed_call_ids_max_size // 10)
            keys_to_remove = list(self._completed_call_ids.keys())[:evict_count]
            for key in keys_to_remove:
                self._completed_call_ids.pop(key, None)
        self._completed_call_ids[litellm_call_id] = True

    def _remember_decremented_call_id(self, litellm_call_id: str) -> None:
        if len(self._decremented_call_ids) >= self._decremented_call_ids_max_size:
            evict_count = max(1, self._decremented_call_ids_max_size // 10)
            keys_to_remove = list(self._decremented_call_ids.keys())[:evict_count]
            for key in keys_to_remove:
                self._decremented_call_ids.pop(key, None)
        self._decremented_call_ids[litellm_call_id] = True

    def _should_decrement(
        self, litellm_call_id: Optional[str], callback_type: str, load_key_source: str
    ) -> bool:
        if not litellm_call_id:
            return True

        if litellm_call_id in self._decremented_call_ids:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                f"callback_type={callback_type}, reason=duplicate_terminal_callback, "
                f"call_id={self._format_call_id(litellm_call_id)}, "
                f"decremented_call_ids={len(self._decremented_call_ids)}"
            )
            self._counter_events.labels("decrement", "skipped_duplicate_terminal", load_key_source).inc()
            return False

        self._remember_decremented_call_id(litellm_call_id)
        return True

    # =========================================================================
    # TTL Refresh
    # =========================================================================

    def _refresh_cache_ttl(self, cache_key: str) -> None:
        """
        Refresh Redis TTL on every increment/decrement.

        The shared redis_cache.increment_cache only sets TTL on first key creation
        (when current_ttl == -1). For sustained traffic lasting > cache_ttl seconds,
        the key would expire and in-flight decrements would hit a fresh key at 0,
        going negative. By refreshing TTL on every access, the key only expires
        after cache_ttl seconds of ZERO activity to that deployment.
        """
        try:
            if (
                self.router_cache.redis_cache is not None
                and hasattr(self.router_cache.redis_cache, "redis_client")
                and self.router_cache.redis_cache.redis_client is not None
            ):
                self.router_cache.redis_cache.redis_client.expire(cache_key, self.cache_ttl)
        except Exception:
            pass  # Best-effort — if Redis is down, we can't refresh TTL anyway

    async def _async_refresh_cache_ttl(self, cache_key: str) -> None:
        """Async variant: refresh Redis TTL on every access."""
        try:
            if self.router_cache.redis_cache is not None:
                _redis_client = self.router_cache.redis_cache.init_async_client()
                await _redis_client.expire(cache_key, self.cache_ttl)
        except Exception:
            pass

    def _reset_negative_counter_if_still_negative(
        self,
        cache_key: str,
        model_group: str,
        dep_id: str,
        new_value: object,
        litellm_call_id: Optional[str],
        callback_type: str,
    ) -> None:
        current_value = self.router_cache.get_cache(key=cache_key, redis_only=True)
        try:
            current_int = int(current_value) if current_value is not None else None
        except (TypeError, ValueError):
            current_int = None

        if current_int is not None and current_int >= 0:
            verbose_router_logger.warning(
                f"[StickyLeastBusyWeighted COUNTER-RESET-SKIP] reason=current_count_non_negative "
                f"callback_type={callback_type}, model_group={model_group}, "
                f"deployment_id={dep_id}, cache_key={cache_key}, "
                f"observed_new_count={new_value}, current_count={current_value}, "
                f"call_id={self._format_call_id(litellm_call_id)}"
            )
            return

        self._counter_resets.labels(model_group, dep_id).inc()
        self.router_cache.set_cache(key=cache_key, value=0, ttl=self.cache_ttl)
        verbose_router_logger.warning(
            f"[StickyLeastBusyWeighted COUNTER-RESET] reason=negative_count "
            f"callback_type={callback_type}, model_group={model_group}, "
            f"deployment_id={dep_id}, cache_key={cache_key}, "
            f"observed_new_count={new_value}, current_count={current_value}, "
            f"reset_to=0, call_id={self._format_call_id(litellm_call_id)}"
        )

    async def _async_reset_negative_counter_if_still_negative(
        self,
        cache_key: str,
        model_group: str,
        dep_id: str,
        new_value: object,
        litellm_call_id: Optional[str],
        callback_type: str,
    ) -> None:
        current_value = await self.router_cache.async_get_cache(key=cache_key, redis_only=True)
        try:
            current_int = int(current_value) if current_value is not None else None
        except (TypeError, ValueError):
            current_int = None

        if current_int is not None and current_int >= 0:
            verbose_router_logger.warning(
                f"[StickyLeastBusyWeighted COUNTER-RESET-SKIP] reason=current_count_non_negative "
                f"callback_type={callback_type}, model_group={model_group}, "
                f"deployment_id={dep_id}, cache_key={cache_key}, "
                f"observed_new_count={new_value}, current_count={current_value}, "
                f"call_id={self._format_call_id(litellm_call_id)}"
            )
            return

        self._counter_resets.labels(model_group, dep_id).inc()
        await self.router_cache.async_set_cache(key=cache_key, value=0, ttl=self.cache_ttl)
        verbose_router_logger.warning(
            f"[StickyLeastBusyWeighted COUNTER-RESET] reason=negative_count "
            f"callback_type={callback_type}, model_group={model_group}, "
            f"deployment_id={dep_id}, cache_key={cache_key}, "
            f"observed_new_count={new_value}, current_count={current_value}, "
            f"reset_to=0, call_id={self._format_call_id(litellm_call_id)}"
        )

    # =========================================================================
    # Streaming Dedup
    # =========================================================================

    def _remember_incremented_call_id(self, litellm_call_id: str) -> None:
        if len(self._seen_call_ids) >= self._seen_call_ids_max_size:
            evict_count = self._seen_call_ids_max_size // 10
            keys_to_remove = list(self._seen_call_ids.keys())[:evict_count]
            for key in keys_to_remove:
                self._seen_call_ids.pop(key, None)
            verbose_router_logger.debug(
                f"[StickyLeastBusyWeighted DEDUP] Evicted {evict_count} old call_ids "
                f"(was at capacity {self._seen_call_ids_max_size})"
            )

        self._seen_call_ids[litellm_call_id] = True

    def _should_increment(self, litellm_call_id: str) -> bool:
        if litellm_call_id in self._seen_call_ids:
            verbose_router_logger.debug(
                f"[StickyLeastBusyWeighted DEDUP] Skipping duplicate increment "
                f"for call_id={litellm_call_id[:16]}... "
                f"(streaming chunk dedup)"
            )
            return False
        if litellm_call_id in self._completed_call_ids:
            verbose_router_logger.debug(
                f"[StickyLeastBusyWeighted DEDUP] Skipping increment for completed "
                f"call_id={litellm_call_id[:16]}..."
            )
            return False

        self._decremented_call_ids.pop(litellm_call_id, None)
        return True

    def _cleanup_call_id(self, litellm_call_id: str, mark_completed: bool) -> None:
        self._seen_call_ids.pop(litellm_call_id, None)
        self._selected_deployments_by_call_id.pop(litellm_call_id, None)
        if mark_completed:
            self._remember_completed_call_id(litellm_call_id)

    # =========================================================================
    # CustomLogger Callbacks - Request Tracking
    # =========================================================================

    def log_pre_api_call(self, model, messages, kwargs):
        """Increment in-flight count. Deduped by litellm_call_id for streaming."""
        if kwargs is None:
            verbose_router_logger.info(
                "[StickyLeastBusyWeighted COUNTER-SKIP] action=increment reason=kwargs_none"
            )
            return
        try:
            (
                model_group,
                dep_id,
                load_key,
                litellm_call_id,
                tracking_source,
                load_key_source,
                route_map_hit,
            ) = self._get_counter_tracking_info(kwargs)
            if model_group is None or dep_id is None or load_key is None:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=increment "
                    f"{self._get_counter_skip_context(kwargs)}, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )
                self._counter_events.labels("increment", "skipped_missing_tracking", load_key_source).inc()
                return

            if tracking_source != "litellm_params" or route_map_hit:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-RECOVER] action=increment "
                    f"source={tracking_source}, route_map_hit={route_map_hit}, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"load_key_source={load_key_source}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )

            stream = kwargs.get("stream", False)
            if litellm_call_id and not self._should_increment(litellm_call_id):
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=increment "
                    f"reason=duplicate_call_id model_group={model_group}, "
                    f"deployment_id={dep_id}, load_key={self._format_load_key(load_key)}, "
                    f"stream={stream}, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"seen_call_ids={len(self._seen_call_ids)}"
                )
                self._counter_events.labels("increment", "skipped_duplicate_call_id", load_key_source).inc()
                return

            cache_key = self._get_request_count_cache_key(model_group, dep_id, load_key)
            new_value = self.router_cache.increment_cache(key=cache_key, value=1, ttl=self.cache_ttl)
            if litellm_call_id and new_value is not None:
                self._remember_incremented_call_id(litellm_call_id)
            if new_value is not None:
                self._refresh_cache_ttl(cache_key)
                self._routing_in_flight.labels(model_group, dep_id).inc()
            counter_result = "applied" if new_value is not None else "cache_increment_returned_none"
            self._counter_events.labels("increment", counter_result, load_key_source).inc()
            previous_count = self._previous_count_from_delta(new_value, 1)
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER] action=increment "
                f"model_group={model_group}, deployment_id={dep_id}, "
                f"load_key={self._format_load_key(load_key)}, "
                f"load_key_source={load_key_source}, "
                f"cache_key={cache_key}, delta=1, previous_count={previous_count}, "
                f"new_count={new_value}, stream={stream}, "
                f"call_id={self._format_call_id(litellm_call_id)}, "
                f"redis_configured={self._is_redis_configured()}, "
                f"seen_call_ids={len(self._seen_call_ids)}"
            )
        except Exception as e:
            verbose_router_logger.error(f"StickyLeastBusy log_pre_api_call error: {e}")

    def _decrement_request_count(self, kwargs, callback_type: str) -> None:
        if kwargs is None:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                f"callback_type={callback_type}, reason=kwargs_none"
            )
            return
        try:
            litellm_call_id = self._extract_litellm_call_id_from_kwargs(kwargs)
            if litellm_call_id and litellm_call_id in self._completed_call_ids:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, "
                    f"reason=duplicate_terminal_callback, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"completed_call_ids={len(self._completed_call_ids)}"
                )
                self._counter_events.labels("decrement", "skipped_duplicate_terminal", "completed_call_id").inc()
                return

            (
                model_group,
                dep_id,
                load_key,
                litellm_call_id,
                tracking_source,
                load_key_source,
                route_map_hit,
            ) = self._get_counter_tracking_info(kwargs)
            if model_group is None or dep_id is None or load_key is None:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, "
                    f"{self._get_counter_skip_context(kwargs)}, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )
                self._counter_events.labels("decrement", "skipped_missing_tracking", load_key_source).inc()
                return

            if tracking_source != "litellm_params" or route_map_hit:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-RECOVER] action=decrement "
                    f"callback_type={callback_type}, source={tracking_source}, "
                    f"route_map_hit={route_map_hit}, model_group={model_group}, "
                    f"deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"load_key_source={load_key_source}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )

            call_id_seen_before_cleanup = (
                litellm_call_id in self._seen_call_ids if litellm_call_id else False
            )
            if not self._should_decrement(litellm_call_id, callback_type, load_key_source):
                return

            if litellm_call_id and not call_id_seen_before_cleanup:
                self._decremented_call_ids.pop(litellm_call_id, None)
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, reason=no_matching_increment, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"load_key_source={load_key_source}, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"seen_call_ids={len(self._seen_call_ids)}"
                )
                self._counter_events.labels("decrement", "skipped_no_matching_increment", load_key_source).inc()
                return

            cache_key = self._get_request_count_cache_key(model_group, dep_id, load_key)
            new_value = self.router_cache.increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            if new_value is not None:
                self._refresh_cache_ttl(cache_key)
                self._routing_in_flight.labels(model_group, dep_id).dec()
                self._counter_events.labels("decrement", "applied", load_key_source).inc()
            previous_count = self._previous_count_from_delta(new_value, -1)
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER] action=decrement "
                f"callback_type={callback_type}, model_group={model_group}, "
                f"deployment_id={dep_id}, load_key={self._format_load_key(load_key)}, "
                f"load_key_source={load_key_source}, cache_key={cache_key}, delta=-1, "
                f"previous_count={previous_count}, new_count={new_value}, "
                f"call_id={self._format_call_id(litellm_call_id)}, "
                f"call_id_seen_before_cleanup={call_id_seen_before_cleanup}, "
                f"redis_configured={self._is_redis_configured()}, "
                f"seen_call_ids={len(self._seen_call_ids)}"
            )
            if new_value is None:
                if litellm_call_id:
                    self._decremented_call_ids.pop(litellm_call_id, None)
                self._counter_events.labels("decrement", "cache_increment_returned_none", load_key_source).inc()
                verbose_router_logger.warning(
                    f"[StickyLeastBusyWeighted COUNTER-WARNING] action=decrement "
                    f"callback_type={callback_type}, reason=cache_increment_returned_none, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"cache_key={cache_key}, call_id={self._format_call_id(litellm_call_id)}, "
                    f"redis_configured={self._is_redis_configured()}"
                )
                return
            elif new_value < 0:
                self._reset_negative_counter_if_still_negative(
                    cache_key=cache_key,
                    model_group=model_group,
                    dep_id=dep_id,
                    new_value=new_value,
                    litellm_call_id=litellm_call_id,
                    callback_type=callback_type,
                )

            if litellm_call_id:
                mark_completed = "SUCCESS" in callback_type
                self._cleanup_call_id(litellm_call_id, mark_completed=mark_completed)
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted DEDUP] action=cleanup "
                    f"callback_type={callback_type}, model_group={model_group}, "
                    f"deployment_id={dep_id}, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"removed={call_id_seen_before_cleanup}, "
                    f"mark_completed={mark_completed}, "
                    f"seen_call_ids={len(self._seen_call_ids)}, "
                    f"completed_call_ids={len(self._completed_call_ids)}, "
                    f"tracked_call_ids={len(self._selected_deployments_by_call_id)}"
                )
        except Exception as e:
            verbose_router_logger.error(f"StickyLeastBusy decrement error: {e}")

    async def _async_decrement_request_count(self, kwargs, callback_type: str) -> None:
        if kwargs is None:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                f"callback_type={callback_type}, reason=kwargs_none"
            )
            return
        try:
            litellm_call_id = self._extract_litellm_call_id_from_kwargs(kwargs)
            if litellm_call_id and litellm_call_id in self._completed_call_ids:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, "
                    f"reason=duplicate_terminal_callback, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"completed_call_ids={len(self._completed_call_ids)}"
                )
                self._counter_events.labels("decrement", "skipped_duplicate_terminal", "completed_call_id").inc()
                return

            (
                model_group,
                dep_id,
                load_key,
                litellm_call_id,
                tracking_source,
                load_key_source,
                route_map_hit,
            ) = self._get_counter_tracking_info(kwargs)
            if model_group is None or dep_id is None or load_key is None:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, "
                    f"{self._get_counter_skip_context(kwargs)}, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )
                self._counter_events.labels("decrement", "skipped_missing_tracking", load_key_source).inc()
                return

            if tracking_source != "litellm_params" or route_map_hit:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-RECOVER] action=decrement "
                    f"callback_type={callback_type}, source={tracking_source}, "
                    f"route_map_hit={route_map_hit}, model_group={model_group}, "
                    f"deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"load_key_source={load_key_source}, "
                    f"call_id={self._format_call_id(litellm_call_id)}"
                )

            call_id_seen_before_cleanup = (
                litellm_call_id in self._seen_call_ids if litellm_call_id else False
            )
            if not self._should_decrement(litellm_call_id, callback_type, load_key_source):
                return

            if litellm_call_id and not call_id_seen_before_cleanup:
                self._decremented_call_ids.pop(litellm_call_id, None)
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted COUNTER-SKIP] action=decrement "
                    f"callback_type={callback_type}, reason=no_matching_increment, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"load_key_source={load_key_source}, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"seen_call_ids={len(self._seen_call_ids)}"
                )
                self._counter_events.labels("decrement", "skipped_no_matching_increment", load_key_source).inc()
                return

            cache_key = self._get_request_count_cache_key(model_group, dep_id, load_key)
            new_value = await self.router_cache.async_increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            if new_value is not None:
                await self._async_refresh_cache_ttl(cache_key)
                self._routing_in_flight.labels(model_group, dep_id).dec()
                self._counter_events.labels("decrement", "applied", load_key_source).inc()
            previous_count = self._previous_count_from_delta(new_value, -1)
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER] action=decrement "
                f"callback_type={callback_type}, model_group={model_group}, "
                f"deployment_id={dep_id}, load_key={self._format_load_key(load_key)}, "
                f"load_key_source={load_key_source}, cache_key={cache_key}, delta=-1, "
                f"previous_count={previous_count}, new_count={new_value}, "
                f"call_id={self._format_call_id(litellm_call_id)}, "
                f"call_id_seen_before_cleanup={call_id_seen_before_cleanup}, "
                f"redis_configured={self._is_redis_configured()}, "
                f"seen_call_ids={len(self._seen_call_ids)}"
            )
            if new_value is None:
                if litellm_call_id:
                    self._decremented_call_ids.pop(litellm_call_id, None)
                self._counter_events.labels("decrement", "cache_increment_returned_none", load_key_source).inc()
                verbose_router_logger.warning(
                    f"[StickyLeastBusyWeighted COUNTER-WARNING] action=decrement "
                    f"callback_type={callback_type}, reason=cache_increment_returned_none, "
                    f"model_group={model_group}, deployment_id={dep_id}, "
                    f"load_key={self._format_load_key(load_key)}, "
                    f"cache_key={cache_key}, call_id={self._format_call_id(litellm_call_id)}, "
                    f"redis_configured={self._is_redis_configured()}"
                )
                return
            elif new_value < 0:
                await self._async_reset_negative_counter_if_still_negative(
                    cache_key=cache_key,
                    model_group=model_group,
                    dep_id=dep_id,
                    new_value=new_value,
                    litellm_call_id=litellm_call_id,
                    callback_type=callback_type,
                )

            if litellm_call_id:
                mark_completed = "SUCCESS" in callback_type
                self._cleanup_call_id(litellm_call_id, mark_completed=mark_completed)
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted DEDUP] action=cleanup "
                    f"callback_type={callback_type}, model_group={model_group}, "
                    f"deployment_id={dep_id}, "
                    f"call_id={self._format_call_id(litellm_call_id)}, "
                    f"removed={call_id_seen_before_cleanup}, "
                    f"mark_completed={mark_completed}, "
                    f"seen_call_ids={len(self._seen_call_ids)}, "
                    f"completed_call_ids={len(self._completed_call_ids)}, "
                    f"tracked_call_ids={len(self._selected_deployments_by_call_id)}"
                )
        except Exception as e:
            verbose_router_logger.error(f"StickyLeastBusy async decrement error: {e}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs, callback_type="SYNC-SUCCESS")
        if self.test_flag:
            self.logged_success += 1

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs, callback_type="SYNC-FAILURE")
        if self.test_flag:
            self.logged_failure += 1

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_decrement_request_count(kwargs, callback_type="ASYNC-SUCCESS")
        if self.test_flag:
            self.logged_success += 1

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_decrement_request_count(kwargs, callback_type="ASYNC-FAILURE")
        if self.test_flag:
            self.logged_failure += 1

    # =========================================================================
    # Load Querying
    # =========================================================================

    def _get_request_counts(self, model_group: str, healthy_deployments: list) -> Dict[str, int]:
        """Sync: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        none_count = 0
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            load_key, load_key_source = self._get_load_key_for_deployment(d)
            cache_key = self._get_request_count_cache_key(model_group, dep_id, load_key)
            count = self.router_cache.get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            normalized_count = max(0, int(count)) if count is not None else 0
            result[dep_id] = normalized_count
            if load_key is not None:
                self._routing_redis_count_by_load_key.labels(load_key, load_key_source).set(normalized_count)
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER-READ] mode=sync "
                f"model_group={model_group}, deployment_id={dep_id}, "
                f"load_key={self._format_load_key(load_key)}, "
                f"load_key_source={load_key_source}, "
                f"cache_key={cache_key}, redis_only=True, "
                f"redis_configured={self._is_redis_configured()}, "
                f"raw_count={count}, normalized_count={normalized_count}"
            )

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                f"[StickyLeastBusyWeighted WARNING] model_group={model_group}, "
                "Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(model_group, "redis_unavailable", "consistent_hashing").inc()
        return result

    async def _async_get_request_counts(self, model_group: str, healthy_deployments: list) -> Dict[str, int]:
        """Async: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        none_count = 0
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            load_key, load_key_source = self._get_load_key_for_deployment(d)
            cache_key = self._get_request_count_cache_key(model_group, dep_id, load_key)
            count = await self.router_cache.async_get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            normalized_count = max(0, int(count)) if count is not None else 0
            result[dep_id] = normalized_count
            if load_key is not None:
                self._routing_redis_count_by_load_key.labels(load_key, load_key_source).set(normalized_count)
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted COUNTER-READ] mode=async "
                f"model_group={model_group}, deployment_id={dep_id}, "
                f"load_key={self._format_load_key(load_key)}, "
                f"load_key_source={load_key_source}, "
                f"cache_key={cache_key}, redis_only=True, "
                f"redis_configured={self._is_redis_configured()}, "
                f"raw_count={count}, normalized_count={normalized_count}"
            )

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                f"[StickyLeastBusyWeighted WARNING] model_group={model_group}, "
                "Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(model_group, "redis_unavailable", "consistent_hashing").inc()
        return result

    # =========================================================================
    # Deployment Selection Core
    # =========================================================================

    @staticmethod
    def _extract_user_id(request_kwargs: Optional[Dict]) -> Optional[str]:
        """
        Extract a user identifier from request kwargs for per-user sticky routing.

        Looks for (in order of preference):
        1. metadata.user_api_key - the API key used for the request
        2. metadata.user_api_key_user_id - the user ID associated with the API key
        3. user - top-level user field

        Returns None if no identifier is found (falls back to message-only hashing).
        """
        if not request_kwargs:
            return None
        metadata = request_kwargs.get("metadata") or {}
        return metadata.get("user_api_key") or metadata.get("user_api_key_user_id") or request_kwargs.get("user")

    def _get_deployment_info(self, deployment: dict) -> str:
        """Helper to extract key deployment info for logging."""
        try:
            dep_id = deployment.get("model_info", {}).get("id", "unknown")
            api_base = deployment.get("litellm_params", {}).get("api_base", "unknown")
            model = deployment.get("litellm_params", {}).get("model", "unknown")
            return f"[id={dep_id}, model={model}, api_base={api_base}]"
        except Exception as e:
            return f"[error extracting info: {e}]"

    @staticmethod
    def _weight_for(deployment: dict) -> float:
        """
        Relative capacity weight for a deployment, read from
        model_info.sticky_weight. Governs both the deployment's share of the
        consistent hash ring and its load tolerance before rebalancing.

        Must come only from static config/DB (never runtime load) so every pod
        derives the same weight and builds an identical ring. Falls back to 1.0
        (equal weight) for legacy/config models without the field, and guards
        against non-numeric or non-positive values.
        """
        raw = (deployment.get("model_info") or {}).get("sticky_weight", 1.0)
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            return 1.0
        return weight if weight > 0 else 1.0

    def _select_deployment(
        self,
        model_group: str,
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
        weights: Dict[str, float] = {}
        load_keys: Dict[str, str] = {}
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            dep_ids.append(dep_id)
            dep_id_to_deployment[dep_id] = d
            weights[dep_id] = self._weight_for(d)
            load_key, _ = self._get_load_key_for_deployment(d)
            if load_key is not None:
                load_keys[dep_id] = load_key

        cached_ring = self._rings.get(model_group)
        previous_dep_ids = frozenset(item[0] for item in cached_ring[0]) if cached_ring is not None else frozenset()
        self._build_hash_ring(model_group, weights)

        # Expose Redis counts to Prometheus so Grafana can show what routing sees
        for did in dep_ids:
            self._routing_redis_count.labels(model_group, did).set(request_counts.get(did, 0))

        # Capacity-normalized load: divide each deployment's raw in-flight count
        # by its weight, so a higher-capacity node is judged less loaded at the
        # same raw count and tolerates proportionally more requests before
        # rebalancing. Redis counts stay raw — only the comparisons normalize.
        eff_load: Dict[str, float] = {did: request_counts.get(did, 0) / weights.get(did, 1.0) for did in dep_ids}

        total_load = sum(request_counts.get(did, 0) for did in dep_ids)
        avg_load = sum(eff_load.values()) / len(dep_ids) if dep_ids else 0
        min_load = min(eff_load.values(), default=0)

        # Avg+min blend on normalized load: catches skewed distributions where
        # avg alone is pulled up by outliers. E.g. loads [50,25,20,7,5] → avg=21.4
        # but min=5, reference=(21.4+5)/2=13.2, so a node at 25 correctly triggers
        # rebalance.
        reference_load = (avg_load + min_load) / 2
        effective_reference = max(reference_load, 1.0)
        threshold_value = self.imbalance_threshold * effective_reference

        current_dep_ids = frozenset(dep_ids)
        for did in previous_dep_ids - current_dep_ids:
            self._deployment_healthy.labels(model_group, did).set(0)
        for did in dep_ids:
            self._deployment_healthy.labels(model_group, did).set(1)
            self._deployment_weight.labels(model_group, did).set(weights[did])
            self._normalized_load.labels(model_group, did).set(eff_load[did])
        self._healthy_deployments_count.labels(model_group).set(len(dep_ids))
        self._reference_load.labels(model_group).set(reference_load)
        self._threshold_load.labels(model_group).set(threshold_value)

        # --- Log node status overview (raw count, weight, and normalized load) ---
        node_summary = ", ".join(
            f"{did}={request_counts.get(did, 0)}"
            f"/w{weights.get(did, 1.0):g}={eff_load[did]:.2f}"
            f"/load_key={self._format_load_key(load_keys.get(did))}"
            for did in dep_ids
        )

        # Calculate imbalance ratio for each deployment (on normalized load)
        imbalance_ratios = []
        for did in dep_ids:
            load = eff_load[did]
            ref = max(reference_load, 1.0)
            ratio = load / ref if ref > 0 else 0
            imbalance_ratios.append(f"{did}={ratio:.2f}")

        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted ROUTING] model_group={model_group}, "
            f"healthy_deployments={len(dep_ids)}, "
            f"deployment_ids={dep_ids}, "
            f"total_in_flight={total_load}, "
            f"avg_norm_load={avg_load:.2f}, "
            f"min_norm_load={min_load:.2f}, "
            f"reference_load={reference_load:.2f}, "
            f"imbalance_threshold={self.imbalance_threshold}, "
            f"loads_per_deployment=[{node_summary}], "
            f"imbalance_ratios=[{', '.join(imbalance_ratios)}]"
        )

        # Try sticky routing
        if sticky_key:
            preferred_id = self._get_deployment_for_key(model_group, sticky_key)
            if preferred_id and preferred_id in dep_id_to_deployment:
                preferred_load = request_counts.get(preferred_id, 0)
                preferred_eff = eff_load.get(preferred_id, 0)
                preferred_weight = weights.get(preferred_id, 1.0)
                current_ratio = preferred_eff / effective_reference if effective_reference > 0 else 0

                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted STICKY-CHECK] model_group={model_group}, "
                    f"sticky_key={sticky_key[:16]}..., "
                    f"preferred_deployment={preferred_id}, "
                    f"preferred_load={preferred_load}, "
                    f"preferred_weight={preferred_weight:g}, "
                    f"preferred_norm_load={preferred_eff:.2f}, "
                    f"effective_reference={effective_reference:.2f}, "
                    f"threshold_value={threshold_value:.2f}, "
                    f"current_imbalance_ratio={current_ratio:.2f}x "
                    f"(threshold_ratio={self.imbalance_threshold}x)"
                )

                if preferred_eff < threshold_value:
                    selected = dep_id_to_deployment[preferred_id]
                    verbose_router_logger.info(
                        f"[StickyLeastBusyWeighted DECISION] STICKY -> model_group={model_group}, "
                        f"deployment_id={preferred_id}, "
                        f"api_base={selected.get('litellm_params', {}).get('api_base', 'unknown')}, "
                        f"model={selected.get('litellm_params', {}).get('model', 'unknown')}, "
                        f"reason=norm_load_{preferred_eff:.2f}_below_threshold_{threshold_value:.2f}, "
                        f"imbalance_ratio={current_ratio:.2f}x"
                    )
                    self._routing_decisions.labels(model_group, preferred_id, "sticky", "consistent_hashing").inc()
                    return selected
                else:
                    verbose_router_logger.info(
                        f"[StickyLeastBusyWeighted STICKY-OVERRIDE] model_group={model_group}, "
                        f"preferred_deployment={preferred_id} OVERLOADED, "
                        f"norm_load={preferred_eff:.2f} exceeds threshold={threshold_value:.2f}, "
                        f"imbalance_ratio={current_ratio:.2f}x > {self.imbalance_threshold}x, "
                        f"falling_back_to=least_busy"
                    )
                    self._routing_decisions.labels(model_group, preferred_id, "override", "consistent_hashing").inc()
            else:
                verbose_router_logger.info(
                    f"[StickyLeastBusyWeighted STICKY-CHECK] model_group={model_group}, "
                    f"sticky_key={sticky_key[:16]}..., "
                    f"preferred_deployment={preferred_id} "
                    f"not_in_healthy_deployments={list(dep_id_to_deployment.keys())}, "
                    f"falling_back_to=least_busy"
                )
        else:
            verbose_router_logger.info(
                f"[StickyLeastBusyWeighted STICKY-CHECK] model_group={model_group}, "
                f"reason=no_sticky_key, "
                f"using=least_busy"
            )

        # Least-busy fallback with random tie-breaking, on capacity-normalized
        # load (min_load already computed above for reference_load) — so a
        # higher-weight node is preferred at the same raw in-flight count.
        min_deployments = [dep_id_to_deployment[did] for did in dep_ids if eff_load[did] == min_load]
        min_dep_ids = [d["model_info"]["id"] for d in min_deployments]

        selected = random.choice(min_deployments) if min_deployments else random.choice(healthy_deployments)
        selected_dep_id = selected["model_info"]["id"]
        if isinstance(selected_dep_id, int):
            selected_dep_id = str(selected_dep_id)

        # Calculate how much less loaded the selected node is compared to average
        load_difference = avg_load - min_load if dep_ids else 0
        load_reduction_pct = (load_difference / avg_load * 100) if avg_load > 0 else 0

        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted DECISION] LEAST-BUSY -> model_group={model_group}, "
            f"deployment_id={selected_dep_id}, "
            f"api_base={selected.get('litellm_params', {}).get('api_base', 'unknown')}, "
            f"model={selected.get('litellm_params', {}).get('model', 'unknown')}, "
            f"selected_load={min_load}, "
            f"avg_load={avg_load:.2f}, "
            f"load_difference_from_avg={load_difference:.2f} ({load_reduction_pct:.1f}% reduction), "
            f"candidates_with_min_load={len(min_deployments)}/{len(dep_ids)}, "
            f"candidate_ids={min_dep_ids}"
        )
        self._routing_decisions.labels(model_group, selected_dep_id, "least_busy", "consistent_hashing").inc()
        return selected

    # =========================================================================
    # Public API - Called by Router
    # =========================================================================

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> dict:
        # Log available healthy deployments at INFO level for production visibility
        healthy_ids = [str(d.get("model_info", {}).get("id", "unknown")) for d in healthy_deployments]
        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted ROUTING-START] (SYNC) model_group={model_group}, "
            f"healthy_deployments_count={len(healthy_deployments)}, "
            f"healthy_deployment_ids={healthy_ids}"
        )
        try:
            request_counts = self._get_request_counts(model_group, healthy_deployments)
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            selected_deployment = self._select_deployment(model_group, healthy_deployments, request_counts, sticky_key)
            self._remember_selected_deployment_for_kwargs(request_kwargs, model_group, selected_deployment)
            return selected_deployment
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusyWeighted ERROR] Routing failed, falling back to " f"random selection: {e}"
            )
            self._routing_fallback.labels(model_group, "error", "consistent_hashing").inc()
            return random.choice(healthy_deployments)

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> dict:
        # Log available healthy deployments at INFO level for production visibility
        healthy_ids = [str(d.get("model_info", {}).get("id", "unknown")) for d in healthy_deployments]
        verbose_router_logger.info(
            f"[StickyLeastBusyWeighted ROUTING-START] (ASYNC) model_group={model_group}, "
            f"healthy_deployments_count={len(healthy_deployments)}, "
            f"healthy_deployment_ids={healthy_ids}"
        )
        try:
            request_counts = await self._async_get_request_counts(model_group, healthy_deployments)
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            selected_deployment = self._select_deployment(model_group, healthy_deployments, request_counts, sticky_key)
            self._remember_selected_deployment_for_kwargs(request_kwargs, model_group, selected_deployment)
            return selected_deployment
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusyWeighted ERROR] Async routing failed, falling back to " f"random selection: {e}"
            )
            self._routing_fallback.labels(model_group, "error", "consistent_hashing").inc()
            return random.choice(healthy_deployments)
