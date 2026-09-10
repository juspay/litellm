import json
from datetime import datetime
from types import SimpleNamespace

import anyio
import anyio.to_process
import anyio.to_thread
import pytest

from litellm.integrations.gcs_bucket import gcs_logger
from litellm.integrations.gcs_bucket import redaction as gcs_redaction
from litellm.integrations.gcs_bucket.gcs_logger import ProductionGCSLogger


@pytest.mark.asyncio
async def test_gcs_prepares_complete_log_in_one_bounded_process_call(monkeypatch):
    process_calls = []
    thread_calls = []

    async def capture_process_worker(callback, value, log_type, *, cancellable, limiter):
        process_calls.append((callback, log_type, cancellable, limiter))
        return callback(value, log_type)

    async def capture_thread_worker(callback, value, *, abandon_on_cancel, limiter):
        thread_calls.append((callback, abandon_on_cancel, limiter))
        return callback(value)

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)
    monkeypatch.setattr(anyio.to_process, "run_sync", capture_process_worker)
    monkeypatch.setattr(anyio.to_thread, "run_sync", capture_thread_worker)

    result = await gcs_logger._prepare_gcs_payload_async(
        {
            "conversation": {"messages": [{"role": "user", "content": "jane.doe@example.com"}]},
            "response": {
                "content": "jane.doe@example.com",
                "reasoning_content": "jane.doe@example.com",
                "thinking_blocks": [{"thinking": "jane.doe@example.com"}],
            },
        },
        "success",
    )

    assert result is not None
    assert "jane.doe@example.com" not in result
    assert result.count("[REDACTED_") == 4
    assert process_calls == [
        (
            gcs_logger._redact_and_serialize_log,
            "success",
            True,
            gcs_logger._GCS_REDACTION_PROCESS_LIMITER,
        )
    ]
    assert thread_calls == [
        (
            gcs_logger._sanitize_for_json,
            True,
            gcs_logger._GCS_SANITIZATION_THREAD_LIMITER,
        )
    ]
    assert gcs_logger._GCS_REDACTION_PROCESS_LIMITER.total_tokens == 1


@pytest.mark.asyncio
async def test_gcs_real_process_worker_redacts_and_serializes(monkeypatch):
    monkeypatch.setenv("GCS_REDACT_PII", "true")
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)

    result = await gcs_logger._prepare_gcs_payload_async(
        {
            "conversation": {"messages": [{"role": "user", "content": "jane.doe@example.com"}]},
            "response": {},
        },
        "success",
    )

    assert result is not None
    assert "jane.doe@example.com" not in result
    assert "[REDACTED_" in result
    assert json.loads(result)["response"] == {}


def test_gcs_error_log_keeps_first_message_json_string(monkeypatch):
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)

    result = gcs_logger._redact_and_serialize_log(
        {"request": {"first_message": [{"role": "user", "content": "jane.doe@example.com"}]}},
        "error",
    )

    error_log = json.loads(result)
    assert isinstance(error_log["request"]["first_message"], str)
    assert "jane.doe@example.com" not in error_log["request"]["first_message"]


def test_gcs_enabled_path_defers_field_sanitization(monkeypatch):
    value = object()

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    assert gcs_logger._sanitize_before_building_log(value) is value

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", False)
    assert gcs_logger._sanitize_before_building_log(value) == str(value)


def test_gcs_backlog_warning_is_rate_limited(monkeypatch):
    warnings = []
    timestamps = iter([100.0, 110.0, 161.0])

    gcs_logger._warn_redaction_backlog_full_once.cache_clear()
    monkeypatch.setattr(gcs_logger.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(gcs_logger.verbose_logger, "warning", lambda message: warnings.append(message))

    gcs_logger._warn_redaction_backlog_full("success")
    gcs_logger._warn_redaction_backlog_full("success")
    gcs_logger._warn_redaction_backlog_full("success")

    assert len(warnings) == 2


@pytest.mark.asyncio
async def test_gcs_redaction_backlog_is_bounded(monkeypatch):
    preparation_started = anyio.Event()
    release_preparation = anyio.Event()
    uploaded_payloads = []
    dropped_log_types = []

    async def blocking_thread_worker(callback, value, *, abandon_on_cancel, limiter):
        preparation_started.set()
        await release_preparation.wait()
        return callback(value)

    async def capture_process_worker(callback, value, log_type, *, cancellable, limiter):
        return callback(value, log_type)

    async def capture_headers(service_account_json, vertex_instance):
        return {}

    async def capture_upload(headers, bucket_name, object_name, logging_payload):
        uploaded_payloads.append(logging_payload)

    monkeypatch.setattr(gcs_logger, "_GCS_INFLIGHT_LOG_LIMITER", anyio.CapacityLimiter(1))
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)
    monkeypatch.setattr(anyio.to_thread, "run_sync", blocking_thread_worker)
    monkeypatch.setattr(anyio.to_process, "run_sync", capture_process_worker)
    monkeypatch.setattr(
        gcs_logger,
        "_warn_redaction_backlog_full",
        lambda log_type: dropped_log_types.append(log_type),
    )
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger.gcs_base, "construct_request_headers", capture_headers)
    monkeypatch.setattr(logger.gcs_base, "_log_json_data_on_gcs", capture_upload)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            logger._upload_to_gcs_async,
            {
                "correlation_id": "first",
                "conversation": {"messages": []},
                "response": {},
            },
            "success-bucket",
            "success",
        )
        await preparation_started.wait()
        await logger._upload_to_gcs_async(
            {
                "correlation_id": "second",
                "conversation": {"messages": []},
                "response": {},
            },
            "success-bucket",
            "success",
        )
        release_preparation.set()

    assert dropped_log_types == ["success"]
    assert len(uploaded_payloads) == 1
    assert json.loads(uploaded_payloads[0])["correlation_id"] == "first"


@pytest.mark.asyncio
async def test_gcs_redaction_slot_is_released_after_worker_failure(monkeypatch):
    inflight_limiter = anyio.CapacityLimiter(1)

    async def capture_thread_worker(callback, value, *, abandon_on_cancel, limiter):
        return callback(value)

    async def fail_process_worker(callback, value, log_type, *, cancellable, limiter):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(gcs_logger, "_GCS_INFLIGHT_LOG_LIMITER", inflight_limiter)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(anyio.to_thread, "run_sync", capture_thread_worker)
    monkeypatch.setattr(anyio.to_process, "run_sync", fail_process_worker)
    logger = ProductionGCSLogger()

    await logger._upload_to_gcs_async(
        {
            "correlation_id": "request-1",
            "conversation": {"messages": []},
            "response": {},
        },
        "success-bucket",
        "success",
    )

    assert inflight_limiter.borrowed_tokens == 0


@pytest.mark.asyncio
async def test_gcs_disabled_path_does_not_start_workers(monkeypatch):
    captured_payload = None

    async def fail_if_called(*args, **kwargs):
        pytest.fail("worker should not run while redaction is disabled")

    async def capture_headers(service_account_json, vertex_instance):
        return {}

    async def capture_upload(headers, bucket_name, object_name, logging_payload):
        nonlocal captured_payload
        captured_payload = logging_payload

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", False)
    monkeypatch.setattr(anyio.to_thread, "run_sync", fail_if_called)
    monkeypatch.setattr(anyio.to_process, "run_sync", fail_if_called)
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger.gcs_base, "construct_request_headers", capture_headers)
    monkeypatch.setattr(logger.gcs_base, "_log_json_data_on_gcs", capture_upload)

    await logger._upload_to_gcs_async(
        {"correlation_id": "request-1", "message": "hello"},
        "success-bucket",
        "success",
    )

    assert captured_payload is not None
    assert json.loads(captured_payload)["message"] == "hello"


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
