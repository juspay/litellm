import json
from pathlib import Path

import pytest

from litellm.integrations.gcs_bucket import gcs_logger


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
