import json
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from litellm.integrations.gcs_bucket import gcs_logger
from litellm.integrations.gcs_bucket.gcs_logger import ProductionGCSLogger


@pytest.mark.asyncio
async def test_gcs_pii_redaction_runs_outside_event_loop(monkeypatch):
    event_loop_thread_id = threading.get_ident()
    redaction_thread_ids = []

    def capture_redaction_thread(messages):
        redaction_thread_ids.append(threading.get_ident())
        return messages

    monkeypatch.setattr(gcs_logger, "redact_messages", capture_redaction_thread)
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)

    logger = ProductionGCSLogger()
    logger.success_bucket_name = "success-bucket"

    async def capture_upload(data, bucket_name, log_type):
        return None

    logger._upload_to_gcs_async = capture_upload
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "jane@example.com"}],
        "litellm_params": {
            "metadata": {"user_api_key_user_email": "owner@example.com"}
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
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
    )

    assert redaction_thread_ids
    assert redaction_thread_ids[0] != event_loop_thread_id


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
