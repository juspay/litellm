# Client-Side Encryption for GCS Logs

Production log events (success + failure) contain conversations, user PII, model
details, and error traces. This feature encrypts every log file **on the proxy,
before it leaves for GCS**, so that anything stored in the bucket is readable
only by whoever can ask KMS to unwrap the file's key — not by the bucket reader
alone, and not even by the proxy that wrote it.

Encryption is **opt-in**: set two environment variables and restart the proxy.
Without them, logging behaves exactly as before (plain JSON uploads).

## What changed

1. **Every uploaded log is encrypted client-side** (new module
   `log_encryption.py`, wired into `gcs_logger.py`).
2. **Encrypted files get a `.enc.json` suffix**, so encrypted vs plain objects
   are distinguishable at a glance in the bucket.
3. **A small readable `metadata` block travels outside the encryption**
   (log type, correlation id, date, timestamp, sanitized user part) — the same
   information the object path/name already exposed, so no new privacy surface.
4. **Filenames include the user**:
   `success/date=<date>/<time>_<user>_<correlation_id>.enc.json`
   (falls back to `unknown` when the key has no user attached).
5. **Fail-closed**: if encryption is switched on but misconfigured, uploads are
   *blocked* — nothing is ever written to GCS in plaintext "temporarily".

## The flow

```
request completes / fails
        │
        ▼
 log record assembled (conversation, response, usage, user, cost, timing…)
        │
        ├─ redaction runs FIRST (if enabled: messages are masked if they
        │  match the configured PII patterns)
        ▼
 JSON serialization
        │
        ▼
 client-side encryption ── always the FINAL step before upload
   ├─ fresh random AES-256 key + fresh nonce  (new for EVERY file)
   ├─ log JSON encrypted with AES-256-GCM
   ├─ the AES key wrapped with the KMS public key (RSA-OAEP + SHA-256)
   ├─ envelope assembled: kms key reference + readable metadata +
   │  wrapped key + nonce + ciphertext
   ▼
 upload to gs://<bucket>/success|failure/date=YYYY-MM-DD/…_<user>_<id>.enc.json
```

## Envelope shape (what sits in the bucket)

| Field            | Readable? | Purpose |
| ---------------- | --------- | ------- |
| `kms`            | yes       | Which KMS key version wraps the file (supports rotation)
| `original_filename` | yes    | The `.json` name the log would have had
| `metadata`       | yes       | Ops lookup fields (log_type, correlation_id, date, timestamp, user)
| `encryption`     | yes       | Algorithm + per-file nonce
| `wrapped_key`    | yes (opaque) | The per-file AES key, encrypted by KMS public key
| `data`           | **no**    | The actual log payload — ciphertext only

## The logic (why it is built this way)

- **Envelope / hybrid encryption**: a fresh random data key per file, wrapped
  by the KMS public key. This is the same construction as KMS envelope
  encryption and GPG/TLS — symmetric speed for data, asymmetric strength for
  key delivery.
- **Fresh key + fresh nonce for every single upload**: nothing is reused, so a
  GCM key/nonce reuse catastrophe is impossible by construction, and no two
  files share anything that relates them cryptographically.
- **The public key is loaded once at proxy startup** and kept in memory.
  Public keys are not secrets — they can only encrypt. Pulling it once and
  caching is the intended operation; there are **no KMS calls during logging**.
- **The proxy can never read back what it wrote.** The private key never
  leaves KMS; decryption tooling unwraps each file's `wrapped_key` via
  `asymmetricDecrypt`. Even a fully compromised proxy or bucket-reader gets
  ciphertext only.
- **Redaction before encryption**: PII masking (when enabled) happens on the
  log record first, encryption last. Real users' `user_email` and similar
  sensitive values are always in ciphertext; only the sanitized user-part
  (which already appeared in filenames) remains readable.
- **Integrity is enforced by AES-GCM**: any tampered byte in an uploaded file
  fails authentication at decrypt time instead of silently altering logs.

## Enabling it

```bash
export GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE=projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>/cryptoKeyVersions/<v>
```

That's it — one variable. At startup the proxy fetches the RSA **public** key
straight from Cloud KMS (`getPublicKey`) using Application Default Credentials
(workload identity on GKE / service account impersonation / local
`gcloud auth application-default login`). No key file is ever deployed or
checked in, and the proxy can never decrypt anything.

Restart the proxy. All loggers read this once at startup.

**Optional offline override** (dev machines / tests without KMS access):

```bash
export GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH=/secure/path/public-key.pem   # PEM takes precedence over the KMS fetch
```

Setting the PEM path without the KMS resource is "misconfigured" — uploads are
blocked (fail-closed), same as a fetch/permission failure.

Existing behaviours preserved: bucket names, folder partitioning
(`success|failure/date=…`), redaction flag, mock mode — all unchanged.

## Rotation

Bump the key version in `GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE` and restart the
proxy — it picks up the new public key automatically. Old files keep
decrypting because the envelope records exactly which key version wrapped
them, and KMS keeps older versions.

## Cost of encryption

- Payload growth ≈ **1.34×** (base64 in the envelope), e.g. a 100 KB log ≈ 135 KB on GCS.
- Latency: single-digit milliseconds for typical logs; ~0.5–0.9 s for a 100 KB log
  through the full proxy upload path (network-bound, encryption is a small part).

## Verified behaviour

End-to-end (real proxy, real `gs://` bucket): success, failure, and big
conversations (~100 KB / ~210 KB) uploaded as `.enc.json`, downloaded, and
decrypted back to **byte-exact** logs — zero data loss; tampered files are
rejected; nothing sensitive exists outside the ciphertext beyond the intended
metadata/filename fields.
