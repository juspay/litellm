import json
import threading
from datetime import datetime
from types import SimpleNamespace

import anyio
import pytest

from litellm.integrations.gcs_bucket import gcs_logger
import litellm.integrations.gcs_bucket.redaction as gcs_redaction
from litellm.integrations.gcs_bucket.gcs_logger import ProductionGCSLogger
from litellm.litellm_core_utils.logging_worker import LoggingWorker


def _response(choices=None):
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o",
        choices=choices or [],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def _success_kwargs(messages=None):
    return {
        "model": "gpt-4o",
        "messages": messages or [{"role": "user", "content": "hello"}],
        "litellm_params": {
            "metadata": {"user_api_key_user_email": "owner@example.com"}
        },
    }


def _success_call_args(messages=None):
    return (
        _success_kwargs(messages),
        _response(),
        datetime.now(),
        datetime.now(),
    )


@pytest.mark.asyncio
async def test_gcs_pii_redaction_runs_outside_event_loop(monkeypatch):
    event_loop_thread_id = threading.get_ident()
    redaction_thread_ids = []

    def capture_redaction_thread(messages):
        redaction_thread_ids.append(threading.get_ident())
        return messages

    monkeypatch.setattr(gcs_logger, "redact_messages", capture_redaction_thread)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)

    await gcs_logger._redact_messages_async(
        [{"role": "user", "content": "jane@example.com"}]
    )

    assert redaction_thread_ids
    assert redaction_thread_ids[0] != event_loop_thread_id


@pytest.mark.asyncio
async def test_large_gcs_redaction_uses_process_worker(monkeypatch):
    process_calls = []

    async def capture_process_worker(callback, value):
        process_calls.append(callback)
        return value

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(
        gcs_logger.anyio.to_process,
        "run_sync",
        capture_process_worker,
    )

    messages = [
        {
            "role": "user",
            "content": "x" * gcs_logger._GCS_PROCESS_THRESHOLD_BYTES,
        }
    ]
    assert await gcs_logger._redact_messages_async(messages) == messages
    assert process_calls == [gcs_logger.redact_messages]


@pytest.mark.asyncio
async def test_gcs_response_sanitization_does_not_block_event_loop(monkeypatch):
    tool_calls = [{"id": "slow-to-serialize"}]
    original_sanitize = gcs_logger._sanitize_for_json
    sanitization_started = threading.Event()
    release_sanitization = threading.Event()
    sanitization_finished = threading.Event()

    def slow_sanitize(value, seen=None):
        if value is tool_calls:
            sanitization_started.set()
            release_sanitization.wait(timeout=0.2)
        result = original_sanitize(value, seen)
        if value is tool_calls:
            sanitization_finished.set()
        return result

    monkeypatch.setattr(gcs_logger, "_sanitize_for_json", slow_sanitize)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)

    logger = ProductionGCSLogger()
    logger.success_bucket_name = None
    response_obj = _response(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="response",
                    reasoning_content=None,
                    tool_calls=tool_calls,
                    function_call=None,
                    thinking_blocks=None,
                    reasoning_items=None,
                ),
                logprobs=None,
                token_ids=None,
                provider_specific_fields=None,
            )
        ]
    )

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                logger.async_log_success_event,
                _success_kwargs(),
                response_obj,
                datetime.now(),
                datetime.now(),
            )
            with anyio.fail_after(1):
                while not sanitization_started.is_set():
                    await anyio.sleep(0)

            assert not sanitization_finished.is_set()
            release_sanitization.set()
    finally:
        release_sanitization.set()


@pytest.mark.asyncio
async def test_gcs_json_serialization_runs_outside_event_loop(monkeypatch):
    event_loop_thread_id = threading.get_ident()
    serialization_thread_ids = []

    def capture_json_thread(data, default):
        serialization_thread_ids.append(threading.get_ident())
        return "{}"

    async def capture_headers(service_account_json, vertex_instance):
        return {}

    async def capture_upload(headers, bucket_name, object_name, logging_payload):
        return None

    monkeypatch.setattr(gcs_logger.json, "dumps", capture_json_thread)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger.gcs_base, "construct_request_headers", capture_headers)
    monkeypatch.setattr(logger.gcs_base, "_log_json_data_on_gcs", capture_upload)

    await logger._upload_to_gcs_async(
        data={"message": "hello"},
        bucket_name="success-bucket",
        log_type="success",
    )

    assert serialization_thread_ids
    assert serialization_thread_ids[0] != event_loop_thread_id


@pytest.mark.asyncio
async def test_gcs_processing_stays_on_callback_thread_when_redaction_is_disabled(
    monkeypatch,
):
    event_loop_thread_id = threading.get_ident()
    processing_thread_ids = []

    def capture_thread(value):
        processing_thread_ids.append(threading.get_ident())
        return value

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", False)
    monkeypatch.setattr(gcs_logger, "_sanitize_for_json", capture_thread)
    monkeypatch.setattr(gcs_logger, "_serialize_for_gcs", capture_thread)
    monkeypatch.setattr(gcs_logger, "_serialize_logprobs", capture_thread)

    await gcs_logger._sanitize_for_json_async({})
    await gcs_logger._serialize_for_gcs_async({})
    await gcs_logger._serialize_logprobs_async({})

    assert processing_thread_ids == [event_loop_thread_id] * 3


@pytest.mark.asyncio
async def test_gcs_completion_cost_does_not_cross_thread_boundary(monkeypatch):
    event_loop_thread_id = threading.get_ident()
    cost_thread_ids = []

    def capture_cost(completion_response):
        cost_thread_ids.append(threading.get_ident())
        return 0

    monkeypatch.setattr(gcs_logger.litellm, "completion_cost", capture_cost)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", False)
    logger = ProductionGCSLogger()
    logger.success_bucket_name = None
    await logger.async_log_success_event(
        kwargs=_success_kwargs(),
        response_obj=_response(),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert cost_thread_ids == [event_loop_thread_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback_name", ["async_log_success_event", "async_log_failure_event"]
)
async def test_gcs_callback_queues_new_log_while_waiting_for_cpu_slot(
    monkeypatch, callback_name
):
    redaction_started = threading.Event()
    release_redaction = threading.Event()
    redaction_calls = 0

    def blocking_redaction(messages):
        nonlocal redaction_calls
        redaction_calls += 1
        redaction_started.set()
        release_redaction.wait(timeout=1)
        return messages

    monkeypatch.setattr(gcs_logger, "redact_messages", blocking_redaction)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    logger = ProductionGCSLogger()
    logger.success_bucket_name = None
    logger.error_bucket_name = None
    callback = getattr(logger, callback_name)

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                callback,
                *_success_call_args(),
            )
            with anyio.fail_after(1):
                while not redaction_started.is_set():
                    await anyio.sleep(0)

            task_group.start_soon(callback, *_success_call_args())
            await anyio.sleep(0.05)
            assert redaction_calls == 1
            release_redaction.set()

        assert redaction_calls == 2
    finally:
        release_redaction.set()


@pytest.mark.asyncio
async def test_gcs_slow_upload_does_not_hold_redaction_slot(monkeypatch):
    uploads_started = 0
    both_uploads_started = anyio.Event()
    release_uploads = anyio.Event()

    async def blocking_upload(
        data, bucket_name, log_type, serialized_data=None
    ):
        nonlocal uploads_started
        uploads_started += 1
        if uploads_started == 2:
            both_uploads_started.set()
        await release_uploads.wait()

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    logger = ProductionGCSLogger()
    logger.success_bucket_name = "success-bucket"
    monkeypatch.setattr(logger, "_upload_to_gcs_async", blocking_upload)

    async with anyio.create_task_group() as task_group:
        for _ in range(2):
            task_group.start_soon(
                logger.async_log_success_event,
                *_success_call_args(),
            )
            with anyio.fail_after(1):
                while uploads_started < 1:
                    await anyio.sleep(0)
        with anyio.fail_after(1):
            await both_uploads_started.wait()
        release_uploads.set()

    assert uploads_started == 2


@pytest.mark.asyncio
async def test_gcs_logging_worker_timeout_keeps_cpu_work_alive(monkeypatch):
    redaction_started = threading.Event()
    release_redaction = threading.Event()
    redaction_calls = 0

    def blocking_redaction(messages):
        nonlocal redaction_calls
        redaction_calls += 1
        redaction_started.set()
        release_redaction.wait(timeout=1)
        return messages

    monkeypatch.setattr(gcs_logger, "redact_messages", blocking_redaction)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    logger = ProductionGCSLogger()
    logger.error_bucket_name = None
    worker = LoggingWorker(timeout=0.01, concurrency=1)

    try:
        worker.ensure_initialized_and_enqueue(
            logger.async_log_failure_event(*_success_call_args())
        )
        with anyio.fail_after(1):
            while not redaction_started.is_set():
                await anyio.sleep(0)
            await worker.flush()

        release_redaction.set()
        with anyio.fail_after(1):
            while gcs_logger._GCS_CALLBACK_LIMITER.borrowed_tokens:
                await anyio.sleep(0)

        assert redaction_calls == 1
    finally:
        release_redaction.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_gcs_callbacks_remain_concurrent_when_redaction_is_disabled(monkeypatch):
    callbacks_started = 0
    both_callbacks_started = anyio.Event()
    release_callbacks = anyio.Event()

    async def blocking_callback(kwargs, response_obj, start_time, end_time):
        nonlocal callbacks_started
        callbacks_started += 1
        if callbacks_started == 2:
            both_callbacks_started.set()
        await release_callbacks.wait()

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", False)
    logger = ProductionGCSLogger()
    monkeypatch.setattr(logger, "_build_success_log", blocking_callback)

    async with anyio.create_task_group() as task_group:
        for _ in range(2):
            task_group.start_soon(
                logger.async_log_success_event,
                {},
                SimpleNamespace(),
                datetime.now(),
                datetime.now(),
            )
        with anyio.fail_after(1):
            await both_callbacks_started.wait()
        release_callbacks.set()

    assert callbacks_started == 2


@pytest.mark.asyncio
async def test_gcs_failure_log_redacts_exception_and_traceback(monkeypatch):
    captured_upload = {}

    async def capture_upload(data, bucket_name, log_type, serialized_data=None):
        captured_upload["data"] = data

    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.setattr(gcs_redaction, "REDACT_ENABLED", True)
    logger = ProductionGCSLogger()
    logger.error_bucket_name = "error-bucket"
    monkeypatch.setattr(logger, "_upload_to_gcs_async", capture_upload)
    kwargs = _success_kwargs()
    kwargs["exception"] = "provider rejected jane@example.com"
    kwargs["traceback_exception"] = "request from jane@example.com"

    await logger.async_log_failure_event(
        kwargs,
        RuntimeError("jane@example.com"),
        datetime.now(),
        datetime.now(),
    )

    serialized_data = json.dumps(captured_upload["data"])
    assert "jane@example.com" not in serialized_data
    assert "[REDACTED_" in serialized_data


@pytest.mark.asyncio
async def test_gcs_success_log_without_user_email_does_not_dump_raw_kwargs():
    logger = ProductionGCSLogger()
    logger.success_bucket_name = "success-bucket"
    logger.error_bucket_name = "error-bucket"

    captured_upload = {}

    async def capture_upload(data, bucket_name, log_type, serialized_data=None):
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
