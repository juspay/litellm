# GCS (Google Cloud Storage) Bucket Logging on LiteLLM Gateway 

This folder contains the GCS Bucket Logging integration for LiteLLM Gateway. 

## Folder Structure 

- `gcs_bucket.py`: This is the main file that handles failure/success logging to GCS Bucket
- `gcs_bucket_base.py`: This file contains the GCSBucketBase class which handles Authentication for GCS Buckets
- `gcs_logger.py`: Production GCS logger used by the proxy callbacks (success/failure buckets)
- `log_encryption.py`: Optional client-side hybrid encryption of logs before upload — see [ENCRYPTION.md](ENCRYPTION.md)

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
# Optional client-side encryption (one variable, key fetched via workload identity):
# export GCS_LOG_ENCRYPTION_KMS_KEY="gcs-log-encryption"   # see ENCRYPTION.md
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
  └── success/date=2026-10-30/06-41-59_jane_doe_chatcmpl-abc123.json
```
**Error Logs:**
```
gs://litellm-error-logs/
  └── failure/date=2026-10-30/06-32-26_unknown_9e6c1a9b-2d0c-4879-a1fe-9c6e6d7cb20f.json
```
Path: `success|failure/date={YYYY-MM-DD}/{HH-MM-SS}_{user}_{correlation_id}.json`
(`user` = sanitized local part of the API-key user's email, `unknown` when none;
with encryption enabled the objects are named `.enc.json` — see ENCRYPTION.md)

## What's Logged

**Success:** Full conversation, response, usage, cost, user info, timing  
**Error:** Error details, user info, model info, request context

Done! 🎉
