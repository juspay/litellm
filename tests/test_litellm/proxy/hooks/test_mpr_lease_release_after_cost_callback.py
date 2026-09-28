"""
Regression tests for the max_parallel_requests ZSET lease leak on failure
paths where an earlier-registered callback rewrites request_data.

Incident (prod, Sep 21, /v1/systemone): every upstream-422 pass-through
request acquired a Redis ZSET lease but released via the legacy
``decrement_fallback`` path with garbage negative counts — proving the
release dispatcher could not find the lease id. Root cause:

1. ``pre_call_hook`` stashes the lease id into the body ``metadata``
   (metadata channels only, no top-level field).
2. ``_init_kwargs_for_pass_through_endpoint`` pops ``metadata`` (a litellm
   param) into ``kwargs["litellm_params"]["metadata"]``.
3. ``post_call_failure_hook`` iterates callbacks in registration order.
   ``_ProxyDBLogger`` is registered BEFORE the MPR limiter, so its failure
   hook runs first and rebuilt ``request_data["litellm_params"]["metadata"]``
   from the (absent) top-level ``metadata`` — destroying the only copy of
   the lease id.

These tests pin the fixed behavior: the lease id must survive the cost
callback's failure hook, and the limiter must release via the ZSET decrement
script rather than falling back to the legacy string-counter decrement.
"""

from unittest.mock import AsyncMock

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    MAX_PARALLEL_REQUESTS_LEASE_ID_FIELD,
    MAX_PARALLEL_REQUESTS_LEASE_KEY_SUFFIX,
    _PROXY_MaxParallelRequestsHandler_v3 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.utils import InternalUsageCache, hash_token


def _build_passthrough_failure_request_payload() -> dict:
    """Replicates the dict ``_build_passthrough_failure_request_payload``
    constructs for a pass-through request whose body carried no client
    ``metadata``: the pre-call stash lives ONLY in
    ``litellm_params.metadata`` because the top-level body ``metadata`` was
    popped into ``litellm_params`` by ``_init_kwargs_for_pass_through_endpoint``.
    """
    return {
        "model": "jev-latest",
        "litellm_params": {
            "is_centralized_redis_cache_incremented": True,
            "metadata": {
                "user_api_key_hash": "hash123",
                MAX_PARALLEL_REQUESTS_LEASE_ID_FIELD: "lease-abc-123",
            },
        },
        "call_type": "pass_through_endpoint",
        "litellm_call_id": "call-1",
    }


@pytest.fixture
def api_key_hash() -> str:
    return hash_token("sk-12345")


@pytest.fixture
def db_logger(monkeypatch) -> _ProxyDBLogger:
    """A ``_ProxyDBLogger`` whose DB write is mocked out so the failure hook
    runs its full metadata rebuild without touching prisma."""
    logger = _ProxyDBLogger()

    async def _noop_update_database(**kwargs):
        return None

    async def _noop_enrich(metadata, **kwargs):
        return metadata

    monkeypatch.setattr(
        _ProxyDBLogger,
        "_should_track_errors_in_db",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        _ProxyDBLogger,
        "_enrich_failure_metadata_with_key_info",
        _noop_enrich,
    )
    return logger


async def _run_cost_callback_failure_hook(
    db_logger: _ProxyDBLogger,
    request_data: dict,
    api_key_hash: str,
) -> None:
    """Run ``_ProxyDBLogger.async_post_call_failure_hook`` with the shared
    ``proxy_logging_obj.db_spend_update_writer`` mocked, so the full hook
    (including the metadata rebuild under test) executes."""
    from litellm.proxy import proxy_server

    update_database = AsyncMock()
    monkeypatch_target = proxy_server.proxy_logging_obj.db_spend_update_writer
    original = monkeypatch_target.update_database
    monkeypatch_target.update_database = update_database
    try:
        await db_logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=Exception("upstream 422"),
            user_api_key_dict=UserAPIKeyAuth(
                api_key=api_key_hash,
                key_alias="test-key",
                max_parallel_requests=10,
            ),
        )
    finally:
        monkeypatch_target.update_database = original


def _find_lease_id(request_data: dict):
    limiter = _PROXY_MaxParallelRequestsHandler.__new__(
        _PROXY_MaxParallelRequestsHandler
    )
    return limiter._get_max_parallel_requests_lease_id_from_dict(request_data)


@pytest.mark.asyncio
async def test_lease_id_survives_cost_callback_metadata_rebuild(
    db_logger, monkeypatch, api_key_hash
):
    """The passthrough lease id must survive the cost callback's
    ``litellm_params.metadata`` rebuild. Before the fix the rebuild seeded
    the dict only from the (absent) top-level ``metadata`` channel,
    destroying the lease id."""
    request_data = _build_passthrough_failure_request_payload()

    async def _noop_update_database(**kwargs):
        return None

    writer = None
    from litellm.proxy import proxy_server

    writer = proxy_server.proxy_logging_obj.db_spend_update_writer
    monkeypatch.setattr(writer, "update_database", AsyncMock(), raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        getattr(proxy_server, "general_settings", {}),
        raising=False,
    )

    await db_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("upstream 422"),
        user_api_key_dict=UserAPIKeyAuth(
            api_key=api_key_hash,
            key_alias="test-key",
            max_parallel_requests=10,
        ),
    )

    assert _find_lease_id(request_data) == "lease-abc-123"


@pytest.mark.asyncio
async def test_release_uses_zset_script_after_cost_callback_ran(
    db_logger, monkeypatch, api_key_hash
):
    """End-to-end ordering: after the cost callback rewrites
    ``litellm_params.metadata``, the MPR limiter's failure hook must still
    release via the ZSET decrement script (ZREM by lease id), NOT fall back
    to the legacy string-counter decrement."""
    from litellm.proxy import proxy_server

    writer = proxy_server.proxy_logging_obj.db_spend_update_writer
    monkeypatch.setattr(writer, "update_database", AsyncMock(), raising=False)

    request_data = _build_passthrough_failure_request_payload()
    request_data["litellm_params"]["metadata"]["user_api_key_hash"] = api_key_hash

    await db_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("upstream 422"),
        user_api_key_dict=UserAPIKeyAuth(
            api_key=api_key_hash,
            key_alias="test-key",
            max_parallel_requests=10,
        ),
    )

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    release_calls = []

    async def mock_decrement_script(*args, **kwargs):
        release_calls.append((kwargs["keys"], kwargs["args"]))
        return [1, 1, 0]

    parallel_request_handler.max_parallel_requests_decrement_script = (
        mock_decrement_script
    )

    await parallel_request_handler.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("upstream 422"),
        user_api_key_dict=UserAPIKeyAuth(
            api_key=api_key_hash,
            key_alias="test-key",
            max_parallel_requests=10,
        ),
    )

    assert release_calls == [
        (
            [f"{{api_key:{api_key_hash}}}:{MAX_PARALLEL_REQUESTS_LEASE_KEY_SUFFIX}"],
            [
                "lease-abc-123",
                parallel_request_handler._get_max_parallel_requests_key_ttl_ms(),
            ],
        )
    ], "release must ZREM the acquired lease id, not take the legacy fallback"
