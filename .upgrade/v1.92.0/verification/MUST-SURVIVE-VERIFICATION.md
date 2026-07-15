# MUST-SURVIVE Verification - v1.92.0 Upgrade

Every runtime customization that survives audit must map to one or more checks here.

## Budget / Spend / Analytics

- [ ] `FREE_MODELS` budget bypass still permits exhausted users to call configured free models.
- [ ] User budget restriction works across multiple proxy instances.
- [ ] Budget equality case blocks when spend is equal to budget if that remains intended behavior.
- [ ] Daily/weekly/monthly budget reset behavior is preserved.
- [ ] Daily user request duration writes are preserved.
- [ ] DAU / WAU / MAU email-based analytics endpoints return counts matching direct SQL checks.
- [ ] User leaderboard returns expected users and totals for the selected time window.
- [ ] Failure rows retain upstream attribution.
- [ ] Failure-log aggregation UI/API still shows expected rows.
- [ ] Spend analytics filters and time ranges match direct SQL checks.

## MPR / Concurrency

- [ ] Multi-instance MPR limiting works with Redis.
- [ ] Zset implementation does not leak active requests after success, failure, timeout, or client disconnect.
- [ ] Redis drift is bounded after a 10-minute concurrency soak.
- [ ] Redis-only load-balancing counters do not silently fall back to in-memory state when Redis is unavailable.
- [ ] Prometheus MPR/concurrency metrics emit expected labels and values.
- [ ] Behavior is correct when MPR is not set for a key.
- [ ] Behavior is correct when a key expires during decrement.

## Routing / Fallbacks

- [ ] Least-busy routing uses Redis, not accidental in-memory fallback.
- [ ] Sticky least-busy routes stable requests to the expected deployment within TTL.
- [ ] `usage-based-routing-v2` honors `redis_only`.
- [ ] Simple-shuffle custom behavior is preserved if still required.
- [ ] Vision fallback routes image requests away from non-vision models.
- [ ] Silent router strips unsupported `stream_options` as expected.
- [ ] Custom model context/public-name behavior survives.
- [ ] `prompt-caching-scope-*` headers are stripped for `v1/messages` Claude Code compatibility.

## GCS / GCP Logging / Logprobs

- [ ] GCS full request/response logging still writes expected object paths and directory structure.
- [ ] `x-litellm-disable-logging: true` prevents GCS writes.
- [ ] GCP stdout logger emits structured JSON with expected fields.
- [ ] GCP/GCS logger client initialization succeeds and emits expected initialization diagnostics.
- [ ] BigQuery large-query path still works.
- [ ] Error logs include full request and client headers where intended.
- [ ] GCS logs include logprobs and token IDs for all supported chat paths.
- [ ] Anthropic-compatible logs still preserve expected content shape.

## Admin / Users / Service Accounts

- [ ] Non-admin user-delete allowlist still permits only configured users.
- [ ] Bulk user update endpoint preserves team-targeting, spend-reset, and email-lookup behavior.
- [ ] Audit logs are written for user/key/team/model create/update/delete operations.
- [ ] Service-account table migrations apply cleanly.
- [ ] Service-account owner creation works.
- [ ] Service-account user deletion flow works.
- [ ] Service-account user creation handles additional columns.
- [ ] Service-account key creation handles additional columns.
- [ ] Grid automation service-account flow works.
- [ ] GPG/requester-related service-account flow works.
- [ ] Slack service-account notifications work without leaking secrets.

## DB / Infra / Reliability

- [ ] Prisma self-heal survives read failures.
- [ ] Read replica initializes on startup and after restarts.
- [ ] Primary-down startup still serves safe reads from read replica if intended.
- [ ] Aiohttp TCP keepalive is enabled and covered by unit test.
- [ ] Upstream stream is cancelled on client disconnect.
- [ ] Redis connection observability emits expected logs/metrics.

## UI / Playground / Build

- [ ] Dashboard builds successfully.
- [ ] Service-account UI flows work.
- [ ] Logs filters, message filter, model filter, and saved filter state work.
- [ ] Viewer/admin-viewer roles can access model-filtered logs where intended.
- [ ] Concurrent requests tab remains visible to admin viewer.
- [ ] Rate-limit/concurrency verification UI shows expected data.
- [ ] Playground booking, seat-selection, retry, and bench-run flows work.
- [ ] GCR image build still overlays local `litellm-proxy-extras` from source over the PyPI wheel if that remains required.
- [ ] Prisma version inside the built container matches the required pinned version.
- [ ] GCR build pipeline pushes the expected image on a throwaway branch.

## Security / RBAC Regression

- [ ] Custom auth still runs upstream common checks.
- [ ] Custom admin/user/service-account endpoints are not exposed to insufficient roles.
- [ ] Service-account endpoints do not bypass org/team/key constraints.
- [ ] Request/logging paths do not leak `api_key`, `master_key`, DB URLs, service-account secrets, or GPG material.
- [ ] MCP/OAuth additions in v1.92.0 do not accidentally grant access through our custom key/team permissions.
