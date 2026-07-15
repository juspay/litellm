# v1.92.0 Upgrade Audit Summary

**Status:** Planning scaffold only. Per-commit audit not yet complete.

## Known Inventory

| Item | Value |
|---|---:|
| Upstream base | `v1.83.3-stable` |
| Upstream target | `v1.92.0` |
| Current Juspay source | `release/v1.83.3` |
| Custom non-merge commits in source range | 142 |
| Upstream commits in target range | 3588 |

## Preliminary Risk Summary

| Area | Risk | Notes |
|---|---|---|
| MPR / concurrency limiter | HIGHEST | Latest custom zset implementation lands in `parallel_request_limiter_v3.py`, which upstream also changed. |
| Service-account flow | HIGHEST | New endpoints, schema fields, Slack/GPG automation, and UI must be reconciled with upstream RBAC/security changes. |
| Budgets / spend | HIGH | Prior custom budget behavior was already risky in the v1.83.3 upgrade; upstream added more budget/fallback/cost behavior. |
| Routing | HIGH | Existing custom routing strategies overlap upstream routing/fallback work and new tag/fallback behavior. |
| GCS/logprobs | HIGH | Touches hot request path, proxy server, common request processing, and GCS logger. |
| UI | HIGH | Upstream has large dashboard migration work between v1.83.3 and v1.92.0. |
| DB/read replica | HIGH | Prisma/read-replica fixes need to be compared with upstream self-heal and startup behavior. |
| Old upgrade artifacts | LOW | Likely do not belong in the runtime replay. |

## Pre-Audit Checklist

- [ ] Regenerate `.upgrade/v1.92.0/replay-matrix.csv`.
- [ ] Mark old upgrade artifact commits as DROP/archive candidates.
- [ ] Split all runtime commits into exactly one batch.
- [ ] Identify upstream equivalents for known backports.
- [ ] Assign reviewers for MPR, service accounts, budgets, routing, and UI.
- [ ] Create per-batch audit docs.
- [ ] Dry-run highest-risk commits against `v1.92.0`.

## High-Risk Dry-Run Candidates

| Commit | Reason |
|---|---|
| `325975bff5` | MPR zset limiter, hot concurrency path. |
| `fa38ff2b42` | Service-account automation, Slack, schema, endpoint registration. |
| `28b1e4c604` | Service-account user/key creation fields and tests. |
| `73479de8d0` | GCS logprobs/token IDs in common request path. |
| `05e41940be` | Additional logprobs path touching proxy server. |
| prior FREE_MODELS/user-budget commits | Budget semantics must survive. |
| sticky least-busy Redis commits | Routing behavior must survive. |
| read-replica/Prisma self-heal commits | Startup and failover behavior must survive. |

## Open Decisions

- [ ] Do we want old `.upgrade` v1.83.3 artifacts preserved in the target branch, archived under `docs/archive/`, or dropped from runtime replay?
- [ ] Should service-account schema migrations be replayed as-is, squashed into a new migration, or reconciled into upstream schema state?
- [ ] Should MPR zset limiter replace upstream behavior wholesale or be re-plumbed into upstream's latest limiter hooks?
- [ ] Which branch receives the final PR: `release/v1.83.3`, a new release branch, or `main`?
- [ ] What is the required sandbox/canary duration for this upgrade?
