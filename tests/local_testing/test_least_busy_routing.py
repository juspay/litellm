#### What this tests ####
#    This tests the router's ability to identify the least busy deployment
#    Tests the direct increment/decrement API (not callbacks)

import asyncio
import os
import random
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

### UNIT TESTS FOR LEAST BUSY DIRECT API ###


def test_increment_request_count():
    """Test that increment_request_count works correctly"""
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"

    result = handler.increment_request_count(model_group, deployment_id)
    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
    assert result == 1
    assert test_cache.get_cache(key=cache_key) == 1


def test_decrement_request_count():
    """Test that decrement_request_count works correctly"""
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "gpt-3.5-turbo"
    deployment_id = "1234"

    # Increment first
    handler.increment_request_count(model_group, deployment_id)
    # Then decrement
    result = handler.decrement_request_count(model_group, deployment_id)
    assert result == 0


@pytest.mark.parametrize("async_test", [True, False])
@pytest.mark.asyncio
async def test_router_get_available_deployments(async_test):
    """
    Tests if 'get_available_deployments' returns the least busy deployment
    """
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    model_group = "azure-model"
    # Set individual cache keys for each deployment (new format)
    request_counts = {"1": 10, "2": 54, "3": 100}
    for deployment_id, count in request_counts.items():
        cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
        if async_test is True:
            await router.cache.async_set_cache(key=cache_key, value=count)
        else:
            router.cache.set_cache(key=cache_key, value=count)

    if async_test is True:
        deployment = await router.async_get_available_deployment(
            model=model_group, messages=None, request_kwargs={}
        )
    else:
        deployment = router.get_available_deployment(model=model_group, messages=None)
    print(f"deployment: {deployment}")
    assert deployment["model_info"]["id"] == "1"

    ## run router completion - assert completion event, no change in 'busy'ness once calls are complete

    router.completion(
        model=model_group,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    # With the new try/finally approach, counts should be decremented immediately
    # (no need to wait for callbacks)
    # The deployment that was picked for the completion call will have been
    # incremented then decremented, so counts should match original
    for deployment_id, expected_count in request_counts.items():
        cache_key = f"deployment:{model_group}:{deployment_id}:request_count"
        actual_count = router.cache.get_cache(key=cache_key)
        assert actual_count == expected_count, f"Expected {expected_count} for {deployment_id}, got {actual_count}"


## Test with Real calls ##


@pytest.mark.asyncio
async def test_router_atext_completion_streaming():
    prompt = "Hello, can you generate a 500 words poem?"
    model = "azure-model"
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ### Call the async calls in sequence, so we start 1 call before going to the next.

    ## CALL 1
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    ## CALL 2
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    ## CALL 3
    await asyncio.sleep(random.uniform(0, 2))
    await router.atext_completion(model=model, prompt=prompt, stream=True)

    # With new format, check individual keys for each deployment
    # Each deployment should have been called once (round-robin like behavior when all start at 0)
    for deployment_id in ["1", "2", "3"]:
        cache_key = f"deployment:{model}:{deployment_id}:request_count"
        count = router.cache.get_cache(key=cache_key)
        # After completion, count should be back to 0 (or 1 if still in flight)
        # Since calls complete sequentially, all should be back to 0
        assert count is None or count == 0 or count == 1, f"Failed. deployment_id={deployment_id} has count={count}"


# asyncio.run(test_router_atext_completion_streaming())


@pytest.mark.asyncio
async def test_router_completion_streaming():
    litellm.set_verbose = True
    messages = [
        {"role": "user", "content": "Hello, can you generate a 500 words poem?"}
    ]
    model = "azure-model"
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "os.environ/OPENAI_API_KEY",
                "rpm": 6,
            },
            "model_info": {"id": 3},
        },
    ]
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    ### Call the async calls in sequence, so we start 1 call before going to the next.

    ## CALL 1
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    ## CALL 2
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    ## CALL 3
    await asyncio.sleep(random.uniform(0, 2))
    await router.acompletion(model=model, messages=messages, stream=True)

    # With new format, check individual keys for each deployment
    for deployment_id in ["1", "2", "3"]:
        cache_key = f"deployment:{model}:{deployment_id}:request_count"
        count = router.cache.get_cache(key=cache_key)
        # After completion, count should be back to 0 (or 1 if still in flight)
        assert count is None or count == 0 or count == 1, f"Failed. deployment_id={deployment_id} has count={count}"


def test_atomic_increment_decrement():
    """
    Test that atomic increment and decrement operations work correctly
    using the new direct API
    """
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"
    deployment_id = "test-deployment"

    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"

    # Increment multiple times
    handler.increment_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 1

    handler.increment_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 2

    handler.increment_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 3

    # Decrement
    handler.decrement_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 2

    # Decrement again
    handler.decrement_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 1

    # Decrement again
    handler.decrement_request_count(model_group, deployment_id)
    assert test_cache.get_cache(key=cache_key) == 0

    # Decrement past 0 should reset to 0 (not go negative)
    result = handler.decrement_request_count(model_group, deployment_id)
    assert result == 0
    count = test_cache.get_cache(key=cache_key)
    assert count == 0, f"Count should be 0, got {count}"


@pytest.mark.asyncio
async def test_async_atomic_increment_decrement():
    """
    Test that async atomic increment and decrement operations work correctly
    using the new direct API
    """
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"
    deployment_id = "test-deployment"

    cache_key = f"deployment:{model_group}:{deployment_id}:request_count"

    # Increment via async API
    await handler.async_increment_request_count(model_group, deployment_id)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 1

    await handler.async_increment_request_count(model_group, deployment_id)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 2

    # Decrement via async API
    await handler.async_decrement_request_count(model_group, deployment_id)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 1

    # Decrement again
    await handler.async_decrement_request_count(model_group, deployment_id)
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 0

    # Decrement past 0 should reset to 0
    result = await handler.async_decrement_request_count(model_group, deployment_id)
    assert result == 0
    count = await test_cache.async_get_cache(key=cache_key)
    assert count == 0, f"Count should be 0, got {count}"


def test_get_least_busy_deployment():
    """
    Test that the least busy deployment is correctly selected
    """
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"

    # Create healthy deployments
    healthy_deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "model-1"}},
        {"model_info": {"id": "dep-2"}, "litellm_params": {"model": "model-2"}},
        {"model_info": {"id": "dep-3"}, "litellm_params": {"model": "model-3"}},
    ]

    # Set request counts: dep-1=5, dep-2=2, dep-3=10
    test_cache.set_cache(key=f"deployment:{model_group}:dep-1:request_count", value=5)
    test_cache.set_cache(key=f"deployment:{model_group}:dep-2:request_count", value=2)
    test_cache.set_cache(key=f"deployment:{model_group}:dep-3:request_count", value=10)

    # Should select dep-2 (least busy with count=2)
    selected = handler.get_available_deployments(
        model_group=model_group,
        healthy_deployments=healthy_deployments,
    )

    assert selected["model_info"]["id"] == "dep-2", f"Expected dep-2, got {selected['model_info']['id']}"


@pytest.mark.asyncio
async def test_async_get_least_busy_deployment():
    """
    Test that the async least busy deployment selection works correctly
    """
    test_cache = DualCache()
    handler = LeastBusyLoggingHandler(router_cache=test_cache)
    model_group = "test-model"

    # Create healthy deployments
    healthy_deployments = [
        {"model_info": {"id": "dep-1"}, "litellm_params": {"model": "model-1"}},
        {"model_info": {"id": "dep-2"}, "litellm_params": {"model": "model-2"}},
        {"model_info": {"id": "dep-3"}, "litellm_params": {"model": "model-3"}},
    ]

    # Set request counts: dep-1=5, dep-2=10, dep-3=1
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-1:request_count", value=5)
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-2:request_count", value=10)
    await test_cache.async_set_cache(key=f"deployment:{model_group}:dep-3:request_count", value=1)

    # Should select dep-3 (least busy with count=1)
    selected = await handler.async_get_available_deployments(
        model_group=model_group,
        healthy_deployments=healthy_deployments,
    )

    assert selected["model_info"]["id"] == "dep-3", f"Expected dep-3, got {selected['model_info']['id']}"


def test_ttl_is_30_minutes():
    """
    Test that the TTL is set to 1800 seconds (30 minutes)
    to handle long-running streaming requests
    """
    assert LeastBusyLoggingHandler.REQUEST_COUNT_TTL == 1800


def test_router_inflight_helpers():
    """
    Test the router's _should_track_inflight and related helper methods
    """
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "test-key",
            },
            "model_info": {"id": "dep-1"},
        },
    ]

    # Test with least-busy routing
    router = Router(
        model_list=model_list,
        routing_strategy="least-busy",
        set_verbose=False,
    )  # type: ignore
    assert router._should_track_inflight() is True

    # Test with simple-shuffle routing (should not track)
    router2 = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        set_verbose=False,
    )  # type: ignore
    assert router2._should_track_inflight() is False
