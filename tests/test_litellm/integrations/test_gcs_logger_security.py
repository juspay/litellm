import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import anyio
import anyio.to_process
import pytest

from litellm.integrations.gcs_bucket import gcs_logger
import litellm.integrations.gcs_bucket.redaction as gcs_redaction
from litellm.integrations.gcs_bucket.gcs_logger import ProductionGCSLogger


@pytest.mark.asyncio
async def test_gcs_redaction_uses_a_bounded_process_worker(monkeypatch):
    process_calls = []

    async def capture_process_worker(callback, value, *, cancellable, limiter):
        process_calls.append((callback, limiter))
        return callback(value)

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)
    monkeypatch.setattr(anyio.to_process, "run_sync", capture_process_worker)

    serialized = await gcs_logger._prepare_gcs_payload_async({"message": "email jane.doe@example.com"})

    assert process_calls == [
        (
            gcs_logger._redact_and_serialize,
            gcs_logger._GCS_REDACTION_PROCESS_LIMITER,
        )
    ]
    assert "jane.doe@example.com" not in serialized
    assert gcs_logger._GCS_REDACTION_PROCESS_LIMITER.total_tokens > 1


@pytest.mark.asyncio
async def test_gcs_upload_redacts_the_complete_log_once(monkeypatch):
    process_calls = 0
    captured_payload = None

    async def capture_process_worker(callback, value, *, cancellable, limiter):
        nonlocal process_calls
        process_calls += 1
        return callback(value)

    async def capture_headers(service_account_json, vertex_instance):
        return {}

    async def capture_upload(headers, bucket_name, object_name, logging_payload):
        nonlocal captured_payload
        captured_payload = logging_payload

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)
    monkeypatch.setattr(anyio.to_process, "run_sync", capture_process_worker)
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger.gcs_base, "construct_request_headers", capture_headers)
    monkeypatch.setattr(logger.gcs_base, "_log_json_data_on_gcs", capture_upload)

    await logger._upload_to_gcs_async(
        {
            "correlation_id": "request-1",
            "conversation": {"messages": [{"content": "jane.doe@example.com"}]},
            "error": {"message": "request from jane.doe@example.com failed"},
        },
        "success-bucket",
        "success",
    )

    assert process_calls == 1
    assert captured_payload is not None
    assert "jane.doe@example.com" not in captured_payload
    assert captured_payload.count("[REDACTED_") == 2


@pytest.mark.asyncio
async def test_gcs_active_upload_survives_caller_timeout(monkeypatch):
    preparation_started = anyio.Event()
    release_preparation = anyio.Event()
    upload_completed = anyio.Event()

    async def blocking_preparation(data):
        preparation_started.set()
        await release_preparation.wait()
        return json.dumps(data)

    async def capture_upload(json_data, bucket_name, gcs_path):
        upload_completed.set()

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_logger, "_prepare_gcs_payload_async", blocking_preparation)
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger, "_upload_serialized_to_gcs", capture_upload)

    task = asyncio.create_task(logger._upload_to_gcs_async({"message": "value"}, "bucket", "success"))
    await preparation_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_preparation.set()
    with anyio.fail_after(1):
        await upload_completed.wait()


@pytest.mark.asyncio
async def test_gcs_timeout_cancels_callback_waiting_for_slot():
    active_started = anyio.Event()
    release_active = anyio.Event()
    waiting_started = anyio.Event()

    async def active_callback():
        active_started.set()
        await release_active.wait()

    async def waiting_callback():
        waiting_started.set()

    async def after_slot(_result):
        return None

    active_task = asyncio.create_task(gcs_logger._run_with_gcs_callback_slot(active_callback, after_slot))
    await active_started.wait()
    with anyio.move_on_after(0.01) as cancel_scope:
        await gcs_logger._run_with_gcs_callback_slot(waiting_callback, after_slot)

    assert cancel_scope.cancel_called
    assert not waiting_started.is_set()
    release_active.set()
    await active_task


@pytest.mark.asyncio
async def test_gcs_success_log_without_user_email_does_not_dump_raw_kwargs():
    logger = ProductionGCSLogger()
    logger.success_bucket_name = "success-bucket"
    logger.error_bucket_name = "error-bucket"

    captured_upload = {}

    async def capture_upload(data, bucket_name, log_type):
        captured_upload["data"] = data
        captured_upload["bucket_name"] = bucket_name
        captured_upload["log_type"] = log_type

    logger._upload_to_gcs_async = capture_upload

    provider_api_key = "sk-provider-secret"
    kwargs_only_secret = "secret-only-in-raw-kwargs"
    kwargs = {
        "model": "gpt-4o",
        "api_key": provider_api_key,
        "messages": [{"role": "user", "content": "training prompt"}],
        "raw_request_body": {"prompt": kwargs_only_secret},
        "litellm_params": {
            "api_key": provider_api_key,
            "metadata": {
                "user_api_key_user_id": "service-account-user",
                "user_api_key_team_alias": "platform",
                "user_api_key_metadata": {"department": "engineering"},
            },
        },
    }
    response_obj = SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )

    await logger.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
    )

    data = captured_upload["data"]
    serialized_data = json.dumps(data, default=str)

    assert captured_upload["bucket_name"] == "success-bucket"
    assert captured_upload["log_type"] == "success"
    assert data["user"]["email"] is None
    assert "litellm_kwargs" not in data
    assert provider_api_key not in serialized_data
    assert kwargs_only_secret not in serialized_data
