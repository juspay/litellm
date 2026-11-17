# GCS (Google Cloud Storage) Bucket Logging on LiteLLM Gateway 

This folder contains the GCS Bucket Logging integration for LiteLLM Gateway. 

## Folder Structure 

- `gcs_bucket.py`: This is the main file that handles failure/success logging to GCS Bucket
- `gcs_bucket_base.py`: This file contains the GCSBucketBase class which handles Authentication for GCS Buckets

## Further Reading
- [Doc setting up GCS Bucket Logging on LiteLLM Proxy (Gateway)](https://docs.litellm.ai/docs/proxy/bucket)
- [Doc on Key / Team Based logging with GCS](https://docs.litellm.ai/docs/proxy/team_logging)


# GCS Logger for LiteLLM Using Callbacks

Logger that saves logs to Google Cloud Storage with organized folder structure.

## Quick Start

### 1. Install Dependencies
```bash
pip install google-cloud-storage
```

### 2. Set Environment Variables
```bash
export GCS_SUCCESS_BUCKET_NAME="litellm-success-logs"
export GCS_ERROR_BUCKET_NAME="litellm-error-logs"
# Optional: export GCS_PATH_SERVICE_ACCOUNT="/path/to/service-account.json"
```

### 3. Configure LiteLLM
```yaml
# config.yaml
litellm_settings:
  callbacks: logging.gcs_logger.logger_instance # path

environment_variables:
  GCS_SUCCESS_BUCKET_NAME: "litellm-success-logs"
  GCS_ERROR_BUCKET_NAME: "litellm-error-logs"
```

### 4. Run
```bash
litellm --config config.yaml
```

## GCS Folder Structure

**Success Logs:**
```
gs://litellm-success-logs/
  └── xyne/xyne-training/siraj/2025-12-09_chatcmpl-abc123.json
```
Path: `department/team/user/{date}_{correlation_id}.json`

**Error Logs:**
```
gs://litellm-error-logs/
  └── gpt-4/2025-12-09_uuid-error.json
```
Path: `model/{date}_{correlation_id}.json`

## What's Logged

**Success:** Full conversation, response, usage, cost, user info, timing  
**Error:** Error details, user info, model info, request context

Done! 🎉

## Independent egress alerts

Configure the alert callback separately from GCS logging:

```yaml
litellm_settings:
  callbacks:
    - litellm.integrations.egress_alerts.alert_logger
```

When upgrading from the alert PR, add this callback even if you keep the GCS
logger callback. The GCS logger no longer schedules alert scans. Remove the GCS
callback and bucket variables if uploads are unwanted. The alert callback does
not require GCS buckets or credentials and ignores `x-litellm-disable-logging`.
Set `LITELLM_ALERTS_ENABLED=false` to disable alerts explicitly.

The scanner checks all chat response choices and tool calls. Flagged calls retain
complete arguments for judgment; if the serialized judge request exceeds 64 KB,
the system emits an unverified alert instead of judging truncated evidence.
`JUSPAY_API_KEY` enables the decisions judge. `XOR_DECISIONS_URL` and
`XOR_DECISIONS_MODEL` select its endpoint and model. An alert is emitted when
maliciousness or unauthorized-transfer probability reaches
`ALERT_JUDGE_THRESHOLD` (default 0.7). Missing or invalid judgments produce an
`unverified` alert. These are suspected actions in model output, not proof that a
tool executed or data left the system.

Application warning logs contain only a generated alert ID, a UUID-validated
request ID for correlation when available, assessment status,
risk scores, rule names, and finding count. They exclude conversation content,
arguments, user identifiers, and destination URLs. The optional
`LITELLM_ALERT_WEBHOOK_URL` receives detailed evidence, including sensitive
content; configure it only for a trusted, access-controlled incident destination.
The judge also receives sensitive evidence and must be an approved service.

Each callback instance permits four simultaneous alert pipelines and up to 64
pending events. CPU-heavy scanning, judge-request serialization, and alert
serialization run in a reusable child process, with at most one CPU job per proxy
process at a time. Input snapshot preparation and application log emission run
in a bounded thread; callback objects are not sent to the child. Judge and
webhook network waits remain asynchronous. This isolates regex execution from
the proxy's GIL, but input snapshotting and process communication still incur
some overhead and share machine resources. Excess events produce a `scan dropped` warning. Duplicate call IDs are
suppressed while pending and for the last 1024 completed requests in that
process. Webhook timeouts, transport errors, HTTP 408/429, and server errors are
retried up to three total attempts with exponential delay. Retries reuse the
alert ID and `Idempotency-Key`; receivers should deduplicate using that ID.
Delivery exhaustion produces a warning. Monitor these warnings alongside alert
summaries. Pending work is in memory and is not durable across process exits.
