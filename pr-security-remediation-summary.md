# Security remediation: litellm-vul SCA findings + echo-scan SAST findings

## Relevant issues

## Summary

This PR clears every actionable finding from the four security audits of the proxy:

1. **litellm-vul Strykea SCA scan (scan-1317)** — software-composition findings against the dependency tree (`fix(deps): clear all litellm-vul SCA findings, raise Python floor to 3.11`)
2. **echo-scan SAST audit (PR #811 report)** — 66 code-level findings (0 critical / 20 high / 36 medium / 4 low / 6 info) (`fix(security): remediate all echo-scan SAST findings`)
3. **litellm-vul-2 Strykea SCA scan (scan-1805, SCA-2026-56)** — a rescan of this branch that catalogued every secondary manifest in the tree: 77 findings (5 critical / 39 high / 23 medium / 10 low) (`fix(deps): clear all litellm-vul-2 scan-1805 findings across secondary manifests`)
4. **echo-scan re-scan (PR #939 report)** — 11 findings remaining after the first SAST pass (5 medium / 6 low) (`fix(security): clear all 11 echo-scan re-scan findings`)

Both audits ran against the wrong or stale base: the SCA scan resolved lockfiles from `origin/main` (a stale dev branch building litellm 1.61.x/1.81.3), and a portion of the SAST findings were already fixed in-tree by prior cherry-picks. This PR fixes everything that genuinely remained open on the fork's release line.

## Changes

### Commit 4: re-scan remediation (echo-scan PR #939 report)

All 11 remaining findings are fixed, each with a regression test that fails if the fix regresses:

1. **ORDER BY injection (spend logs):** every sort key now maps to a pinned SQL expression through `_SORT_COLUMN_TO_ORDER_BY_EXPR` in `spend_management_endpoints.py`; the same map validates `sort_by` at request time, so no request-supplied text can reach the ORDER BY clause.
2. **Weak at-rest key derivation:** AES-GCM keys derived with HKDF-SHA256 bound to a purpose info string instead of a single SHA-256 pass. New ciphertexts carry a `v3:hkdf:` marker; the format-detecting decrypt still reads `v2:gcm:` and legacy values, so existing deployments keep working.
3. **Path traversal (plugin proxy):** the captured path is resolved with `posixpath.normpath`, so `/plugin-proxy/<name>/../../admin` cannot climb out of the plugin's configured base URL.
4. **SSRF + runaway guardrail code:** optional `CUSTOM_CODE_HTTP_ALLOWED_HOSTS` egress allowlist (exact host match, enforced before any connection opens) and a 5s `asyncio.wait_for` deadline around `apply_guardrail` so a pathological guardrail cannot pin a worker.
5. **Weak plugin session key:** `_plugin_fernet` now raises when `LITELLM_SALT_KEY` is unset instead of deriving a publicly-computable HMAC(b'', name) key.
6. **SQL injection (pg_views):** the `DATABASE_SCHEMA` value reaches `query_raw` as a bound positional parameter, never interpolated into the SQL string.
7. **Host-header poisoning (login/SSO redirects):** the Host header is trusted only when the caller is on an internal network, a configured trusted proxy, or the Host is in `ALLOWED_REDIRECT_HOSTS`; otherwise the redirect falls back to an origin-relative URL.
8. **SQL injection (tag analytics):** `tag_filter`/`tag_filters` values are bound as positional parameters at all four query sites (`$N` with the parameter appended, never string-interpolated).
9. **Arbitrary command execution (MCP stdio):** relative paths are rejected at pydantic validation time (they resolve against the proxy CWD), and at spawn time the binary is resolved via PATH and refused if it sits in a world-writable directory.
10. **Missing cookie flags:** every login path (`/login`, `/v2/login`, `/v3/login/exchange`, SSO callback) now sets the UI session cookie through `set_ui_session_cookie` — Secure (unless plain HTTP), SameSite=lax, Max-Age matching `LITELLM_UI_SESSION_DURATION`, and HttpOnly opt-in via `LITELLM_UI_SESSION_COOKIE_HTTPONLY` (off by default because the dashboard's reverse-proxy fallback reads the token from document.cookie).
11. **Unverified SSO identity:** id_tokens used for identity now verify against the provider's JWKS (`GENERIC_JWKS_ENDPOINT`, Microsoft tenant endpoint) with kid matching, RS256, and iss/aud/exp checks; the Entra ID app-role mapping fails closed to `[]` on any error.

### Commit 5: inherited lint cleanup

The branch base (`release/v1.92.0`) carried 57 `ruff check` errors that block `make pre-commit` regardless of any change. They are cleared here so CI-style lint passes on this branch: intentional stdout metric/diagnostic prints are marked `noqa: T201` (the parallel-request and service-event log lines are scraped by GCP Cloud Logging and must stay on stdout), unused imports and dead local assignments are removed, the dead first `async_post_call_failure_hook` definition is dropped (Python binds the second, so runtime behavior is unchanged), and stale noqa directives are stripped.

### Commit 3: secondary-manifest remediation (SCA / litellm-vul-2 scan-1805)

The rescan confirmed this branch is what it scanned, and that the first commit had covered the production lockfiles but not the secondary manifests it also catalogued. All 77 findings plus advisories published after the Oct 28 scan snapshot are now clear:

- `.github/scripts/triage-requirements.txt` regenerated with anyio 4.15.1 (critical GHSA-82r6 TLSStream IDNA spoof) and certifi 2026.10.19 (GHSA-6q48, compromised emSign root). This file runs in privileged triage workflows, so the pre-patch certifi meant the triage tooling itself still trusted the compromised root
- Cookbook litellm pin 1.83.14 -> 1.106.5 with transitive floors (anyio, idna, filelock pinned below the advisory floor and inside litellm's cap). The old pin told users to install a release with six open litellm advisories including the critical host-header auth bypass
- litellm-rust: pyo3 0.23.5 -> 0.29.3 (out-of-bounds read GHSA-36hh) with the pyo3 0.26 GIL migration (with_gil -> attach, allow_threads -> detach) in the python-bridge; rustls 0.23.46; h2, quinn-proto, tokio, webpki-roots refreshed for post-snapshot advisories. Full workspace passes fmt, clippy -D warnings on all three crates, and 75 tests
- npm test/dev lockfiles (root, pass_through_tests, ui_unit_tests): js-yaml 3.15.2, brace-expansion 1.1.21/5.0.12, browserslist 4.28.7+, baseline-browser-mapping 2.11+, handlebars 4.7.10 (two criticals), form-data 4.0.6, ws 8.21.0+, protobufjs 7.6.5+, @babel/core 7.29.6+, and deepmerge held at 4.1.2 via overrides (no fixed release exists; 4.1.2 is below the affected range)
- Ruby passthrough Gemfile.lock: faraday 2.8.1 -> 2.14.3

osv-scanner now reports no issues on every manifest in the tree. The one exception is Cargo.lock's block-buffer 0.10.4: unfixable until some axum release supports a tungstenite that drops the sha1 0.10 chain, not part of the scan findings, and requires a caught-panic-then-reuse pattern to exploit

### Commit 1: dependency remediation (SCA / litellm-vul scan-1317)

- **Python floor 3.10 -> 3.11.** CI runs 3.12, the runtime image is 3.13, dev is 3.11; nothing supported 3.10, but its marker kept `filelock` on 3.29, `pillow` on 12.3, and blocked the `dnspython` 2.9.0 floor (GHSA-45c3).
- **Security floors for the open transitive deps** from the scan analysis: `tornado`, `aiohttp`, `anyio`, `urllib3`, `pyjwt`, `sqlparse`, `mako`, `pyasn1`, `fsspec`, `dnspython`, `multidict`, `filelock`, `pypdf`, `h2`, `httplib2`, `setuptools`, `soupsieve`, `werkzeug`, `graphql-core`, `restrictedpython`, `oauthlib`, `langgraph-sdk`; `cryptography` override to 50.x (Bleichenbacher-oracle and subgroup-attack fixes, with `pgpy 0.6.0` capping at `<49`).
- **next 16.2.12 -> 16.4.1**, clearing 11 npm advisories including three 9.5-criticals (GHSA-2464, GHSA-673q, GHSA-vcvr). 16.4.1 type-checks the whole tsconfig project during build while 16.2.x only checked the app graph, so `src/**/*.test.*` are excluded from the production tsconfig; vitest still owns them.
- **pylint 4.0.5 -> 4.0.9** (GHSA-22qq); `lunary` pinned to 1.4.37; `logfire`/`traceloop-sdk`/`claude-agent-sdk` refreshed.
- **GHSA-7hp6-4w63-5g45 (CVE-2026-114612, CVSS 9.9)**: session tokens minted by `get_cli_jwt_auth_token` now use `encrypt_bearer_token` with the `litellm_login_` prefix as AES-256-GCM AAD, so management-route encrypted values are no longer replayable as bearer tokens. Cherry-picked from upstream d098b02ed9 (released in 1.100.4).
- The residual version-string matches (GHSA-7hp6, GHSA-g5ff) and `diskcache` (GHSA-w8v5, no fix published) are documented ignores in `osv-scanner.toml`; both fixes are already in-tree as cherry-picks.

`osv-scanner` now reports no issues against the CI invocation in `.github/workflows/osv-scan.yml`.

### Commit 2: SAST remediation (echo-scan report)

**SSRF / credential exfiltration (high):**
- `token_counter` image-dimension lookup: streamed URL downloads and base64 payloads capped at `MAX_IMAGE_URL_DOWNLOAD_SIZE_MB`, fetched via `safe_get`, falling back to default dimensions on truncated or oversized input instead of parsing attacker-supplied bytes.
- anthropic/azure_ai `count_tokens` handlers: credential destination validated against the provider allowlist before fetching `api_base`; full secrets no longer logged in error paths.
- `sticky_least_busy_weighted` and the `scan_keywords` webhook: non-http(s) schemes rejected before any fetch.
- Request parameter validation and provider URL destination guard (upstream #34189, cherry-picked): route allowlist checks through `auth_utils` and move parameter validation to pre-call.

**Authentication (high):**
- `ui_sso`: `alg=none` and missing-alg JWTs rejected in the unverified decode used for routing (issues #6-#8 in the report).

**Injection (high):**
- `xml` -> `defusedxml` in `factory.py`, `get_blog_posts.py`, `run_llm_translation_tests.py`, `_coverage_check.py` (issues #10-#11 and related).

**Information disclosure (high):**
- Custom secret manager (`my_secret_manager.py`): secret values no longer logged; only names/metadata (issue #4).
- `secret_detection`: request body scanned in memory instead of being written to a temp file first.

**Container/IaC hardening:**
- Containers run as non-root: `Dockerfile`, `ui/Dockerfile`, `prod_entrypoint.sh`, `build_admin_ui.sh` (issues #5, "Last User Is Root", "Missing User").
- Helm: hardcoded default Postgres password and master-key defaults dropped from `secret-dbcredentials.yaml` and `secret-masterkey.yaml` (issue #3).
- Terraform: GCS access-logging bucket wired to both storage buckets; CloudSQL `require_ssl` on writer and reader; ALB policy bumped to `ELBSecurityPolicy-TLS13-1-2-Res-2021-06`.
- `build-gcr` workflow hardening; mock_llm_server CORS scoped to localhost dev origins instead of `*`.

**Other:**
- `md5 -> sha256` in sticky-hash rings and spend-log id generation; hashes are runtime-generated identifiers, so no persisted data depends on md5.
- `s3_v2`: extracted `_virtual_hosted_style_url`, scheme handled via `urlparse` with exact legacy semantics.
- `npm` min-release-age 3 -> 7 in both `.npmrc` files; `uv` exclude-newer cooldown added to `litellm-proxy-extras` and `enterprise` pyprojects.
- `truncate_error_str` imported where exception paths referenced it bare; removed abandoned duplicate `top_key_view` test; ratcheted eslint suppressions down.

## How the findings map to the audits

### litellm-vul (Strykea scan-1317)

The scan resolved versions from `origin/main`, which is a stale dev branch building litellm 1.61.x/1.81.3 — not the deployed release line. A large share of its findings (22 litellm application advisories, plus `python-multipart`, `orjson`, `google-cloud-aiplatform`, `starlette`, `certifi`, etc.) were therefore **already fixed** on `release/v1.92.0`. The genuinely open items were transitive Python deps whose fixed versions postdate the branch's lock; this PR pulls them in via the floor bumps listed above. The `next` bump clears the npm-side advisories (build-time only; the runtime image never copies `node_modules`).

### litellm-vul-2 (Strykea scan-1805, SCA-2026-56)

This rescan targeted `litellm_security_fixes` directly and confirmed the earlier remediation, but it also catalogued every secondary manifest in the tree: the triage workflow's hash-pinned requirements, the cookbook example pin, the Rust workspace's Cargo.lock, a Ruby Gemfile.lock, and three test/dev npm lockfiles. Those are the surfaces commit 3 clears. Notably its two criticals (anyio TLSStream IDNA spoof, certifi emSign root) sat in the triage tooling's own pins, which run with elevated privileges.

### echo-scan SAST report

All 66 findings are addressed: the SSRF cluster (image URL fetch, api_base fetch, scheme validation), the four `verify_signature: False` JWT decode sites in `ui_sso.py`/`handle_jwt.py` (now alg-restricted; the remaining unverified decodes are routing-only and reject `alg=none`/missing-alg), the XXE-prone `xml.etree` imports, the secret-value logging in the custom secret manager, the temp-file handling in `secret_detection`, root-user containers in all three Dockerfiles, hardcoded Helm secret defaults, and the terraform TLS/SSL gaps.

### echo-scan re-scan (PR #939 report)

A re-scan after the first SAST pass found 11 findings still open (5 medium / 6 low); commit 4 clears all of them with the fixes itemized above.

## Pre-Submission checklist

- [x] Meaningful tests added (token_counter download-cap tests, url_utils scheme tests, auth_utils allowlist tests, ui_sso alg tests, azure/anthropic count_tokens destination-guard tests, provider_url_destination_guard tests, plugin-proxy traversal and salt fail-closed tests, MCP stdio validation tests, session-cookie flag tests, host-trust redirect tests, tag-filter parameter-binding tests, ORDER BY allowlist tests, JWKS verification tests including forged-signature and wrong-audience fail-closed cases, HKDF derivation tests, custom-code allowlist and timeout tests)
- [ ] CI/CD checks pass
- [x] Scope isolated: five commits, one per audit plus one lint cleanup
- [ ] Greptile review requested

## Type

🐛 Bug Fix
🚄 Infrastructure
✅ Test
