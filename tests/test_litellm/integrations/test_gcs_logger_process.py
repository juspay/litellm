import json
from pathlib import Path

import pytest

from litellm.integrations.gcs_bucket import gcs_logger


def test_gcs_content_size_accepts_surrogate_code_points():
    # Given
    content = "prefix-\udcff-suffix"

    # When
    size = gcs_logger._string_content_size(content)

    # Then
    assert size > 0


@pytest.mark.asyncio
async def test_large_gcs_redaction_process_removes_pii(monkeypatch):
    monkeypatch.setenv("GCS_REDACT_PII", "true")
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.chdir(Path(__file__).resolve().parents[3])

    messages = [
        {
            "role": "user",
            "content": "x" * gcs_logger._GCS_PROCESS_THRESHOLD_BYTES
            + " jane.doe@example.com",
        }
    ]

    redacted = await gcs_logger._redact_messages_async(messages)

    assert "jane.doe@example.com" not in json.dumps(redacted)


@pytest.mark.asyncio
async def test_large_gcs_redaction_process_accepts_surrogate_code_points(monkeypatch):
    # Given
    monkeypatch.setenv("GCS_REDACT_PII", "true")
    monkeypatch.setattr(gcs_logger, "REDACT_ENABLED", True)
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    messages = [
        {
            "role": "user",
            "content": "x" * gcs_logger._GCS_PROCESS_THRESHOLD_BYTES + "\udcff",
        }
    ]

    # When
    redacted = await gcs_logger._redact_messages_async(messages)

    # Then
    assert redacted == messages
