# Batch Plan - v1.92.0 Upgrade

This is the initial batch model. Do not treat it as final until each commit in `replay-matrix.csv` is assigned exactly once.

| Batch | Name | Candidate commit themes | Default stance |
|---|---|---|---|
| 00 | upgrade-artifacts | old `.upgrade/*`, old `docs/v1.83.3-upgrade-strategy.md`, replay-note commits, old verification reports | DROP from runtime replay or archive intentionally |
| 01 | claude-anthropic-compat | prompt-caching-scope header drops, reasoning field/content-block fixes, Claude Code compatibility | REWORK where upstream Responses/Messages changed |
| 02 | build-ci-docker-prisma | GCR build, Dockerfile changes, Prisma version/pins, proxy-extras overlay, playground build wiring | REWORK after checking upstream Docker/Helm split |
| 03 | gcs-gcp-logging-logprobs | GCP stdout logger, GCS request/response logging, disable header, BigQuery, logprobs/token IDs | HIGH REWORK |
| 04 | budgets-spend-analytics | FREE_MODELS, user budget, spend/failure logs, request duration, daily spend analytics | HIGH REWORK |
| 05 | rate-limit-mpr-concurrency | Redis counters, MPR drift/leak fixes, zset limiter, Prometheus MPR metrics | HIGHEST REWORK |
| 06 | routing-fallbacks | least-busy, sticky least-busy, redis_only, simple-shuffle, vision fallback, silent router | HIGH REWORK |
| 07 | admin-users-service-accounts | user delete allowlist, bulk update, audit logging, service-account CRUD/automation/GPG/Slack | HIGHEST REWORK |
| 08 | ui-dashboard | logs filters, analytics UI, service-account UI, playground UI, package-lock changes | HIGH REWORK |
| 09 | db-read-replica-infra | Prisma self-heal, read replica init, aiohttp keepalive, Redis connection observability | HIGH REWORK |
| 10 | security-rbac-mcp-audit | custom auth/RBAC interactions with v1.92 security hardening, MCP route/auth additions | Audit gate; may not map to one commit group |

## Initial Replay Preference

Replay should start with isolated, low-coupling batches and leave shared hot paths for later:

1. 00 only if we decide to archive docs; otherwise skip.
2. 02 build/CI pieces that are still needed.
3. 09 infra reliability pieces that apply cleanly.
4. 03 logging/logprobs.
5. 01 Claude compatibility.
6. 04 budgets/spend.
7. 07 service accounts/admin.
8. 06 routing.
9. 05 MPR concurrency.
10. 08 UI after API surfaces settle.

For actual replay, preserve chronological order within each batch unless the audit explicitly says otherwise.
