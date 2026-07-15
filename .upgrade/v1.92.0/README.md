# v1.92.0 Upgrade Working Artifacts

Planning artifacts for the `release/v1.83.3` -> upstream `v1.92.0` upgrade.

Main strategy doc: `docs/v1.92.0-upgrade-strategy.md`

## Layout

```text
.upgrade/v1.92.0/
├── README.md
├── batches/
│   └── README.md
├── audit/
│   └── SUMMARY.md
└── verification/
    └── MUST-SURVIVE-VERIFICATION.md
```

## Current Facts

- Planning branch: `v1.92.0-upgrade-plan-from-v1.83.3-stable`
- Runtime source branch: `release/v1.83.3`
- Upstream target tag: `v1.92.0`
- Custom commit inventory: `142` non-merge commits in `v1.83.3-stable..release/v1.83.3`
- Upstream delta size: `3588` commits in `v1.83.3-stable..v1.92.0`

## Generate Replay Matrix

Run this after syncing `release/v1.83.3`:

```bash
git log --reverse --no-merges --format="%H|%ad|%s" --date=short \
  v1.83.3-stable..release/v1.83.3 \
  > .upgrade/v1.92.0/replay-matrix.csv
```

Do not treat the replay matrix as the replay list directly. First classify:

- old `.upgrade` and prior upgrade planning commits
- runtime custom commits
- backports already present in `v1.92.0`
- true no-op/revert pairs

## Audit Record Format

```markdown
## <short-sha> - <subject>
- files:
- intent:
- upstream overlap:
- decision: KEEP-AS-IS | REWORK | DROP | CONDITIONAL DROP
- rationale:
- replay plan:
- verification:
- reviewer:
```

## Suggested Review Order

1. Batch 05: rate-limit / MPR concurrency
2. Batch 07: admin / users / service accounts
3. Batch 04: budgets / spend / analytics
4. Batch 06: routing / fallbacks
5. Batch 03: GCS / GCP logging / logprobs
6. Batch 09: DB/read-replica/infra reliability
7. Batch 08: UI/dashboard
8. Batch 01 and 02: compatibility and build/CI
9. Batch 00: old upgrade artifacts

The replay order can differ from audit order; audit the riskiest first, replay the least coupled first.
