# Security remediation: litellm-vul SCA findings + echo-scan SAST findings

## Relevant issues

## Summary

This PR clears every actionable finding from the ten security audits of the proxy:

1. **litellm-vul Strykea SCA scan (scan-1317)** — software-composition findings against the dependency tree (`fix(deps): clear all litellm-vul SCA findings, raise Python floor to 3.11`)
2. **echo-scan SAST audit (PR #811 report)** — 66 code-level findings (0 critical / 20 high / 36 medium / 4 low / 6 info) (`fix(security): remediate all echo-scan SAST findings`)
3. **litellm-vul-2 Strykea SCA scan (scan-1805, SCA-2026-56)** — a rescan of this branch that catalogued every secondary manifest in the tree: 77 findings (5 critical / 39 high / 23 medium / 10 low) (`fix(deps): clear all litellm-vul-2 scan-1805 findings across secondary manifests`)
4. **echo-scan re-scan (PR #939 report)** — 11 findings remaining after the first SAST pass (5 medium / 6 low) (`fix(security): clear all 11 echo-scan re-scan findings`)
5. **echo-scan re-scan (PR #1048 report)** — 11 findings remaining after the second SAST pass (0 critical / 2 high / 6 medium / 3 low) (`fix(security): clear all 11 echo-scan PR #1048 findings`)
6. **echo-scan re-scan (PR #1492 report)** — 12 findings remaining after the third SAST pass (0 critical / 0 high / 8 medium / 4 low) (`fix(security): clear all 12 echo-scan PR #1492 findings`)
7. **echo-scan re-scan (PR #1511 report)** — 8 findings remaining after the fourth SAST pass (0 critical / 1 high / 4 medium / 3 low) (`fix(security): clear all 8 echo-scan PR #1511 findings`)
8. **echo-scan re-scan (PR #1527 report)** — 10 findings remaining after the fifth SAST pass (0 critical / 0 high / 7 medium / 3 low) (`fix(security): clear all 10 echo-scan PR #1527 findings`)
9. **echo-scan re-scan (PR #1548 report)** — 10 findings remaining after the sixth SAST pass (0 critical / 0 high / 6 medium / 4 low) (`fix(security): clear all 10 echo-scan PR #1548 findings`)

Both audits ran against the wrong or stale base: the SCA scan resolved lockfiles from `origin/main` (a stale dev branch building litellm 1.61.x/1.81.3), and a portion of the SAST findings were already fixed in-tree by prior cherry-picks. This PR fixes everything that genuinely remained open on the fork's release line.

## Changes

### Commit 9: fifth re-scan remediation (echo-scan PR #1527 report)

All 10 findings are fixed, each with a regression test that fails if the fix regresses:

1. **Guardrail custom-code regex/JSON primitives unbounded (CWE-400, low):** every `regex_*` primitive in the custom-code guardrail now caps its inputs (`_MAX_REGEX_TEXT_CHARS=1_000_000` characters of text, `_MAX_REGEX_PATTERN_CHARS=1_000` characters of pattern) and `json_parse` caps its input at `_MAX_JSON_CHARS=1_000_000`; an oversized input is refused per primitive (match/match_all return `False`, replace returns the text unchanged, find_all returns `[]`, json_parse returns `None`) instead of being handed to the regex engine or the JSON parser, so a pathological pattern or payload cannot pin the CPU (both execution paths already carried 5s deadlines, which bound wall-clock but not CPU burn)
2. **Plugin session claim mintable by the plugin itself (CWE-863, low):** the token issued to a plugin is now `<fernet-ciphertext>.<base64url-mac>` where the MAC is HMAC-SHA256 under a proxy-side issuer key derived as `HMAC(LITELLM_SALT_KEY, "litellm-plugin-claim-issuer-v1:"+plugin_name)`; the plugin only ever receives its per-plugin Fernet key, which is a different derivation, so it can decrypt claims but cannot mint one; `verify_plugin_session_claim` checks the MAC first and rejects a claim not proxy-issued before any decryption or role lookup, and the live role is still re-resolved from the DB on every verify
3. **Logo upload stores polyglot bytes verbatim (CWE-434, low):** `/upload/logo` re-encodes the decoded image through Pillow to PNG and stores only the re-encoded pixels, so appended scripts, EXIF payloads, and polyglot suffixes never reach disk; an upload that fails Pillow decode returns 400 (`Invalid image file: it could not be decoded as an image.`) after the existing magic-byte and 5MB checks; served uploaded assets (`/get_image` with a custom logo, `/get_favicon` with a custom favicon) now carry `Content-Disposition: attachment`, so a stored polyglot cannot be rendered as a document on direct navigation (the dashboard `<img>` embed is unaffected)
4. **Skills sandbox accepts traversal-shaped file paths (CWE-22, medium):** bundled skill files are validated with `PurePosixPath` before any host-side path join: absolute paths, `..` segments, backslashes (which are not POSIX separators but are accepted by the host filesystem), NUL bytes, and empty paths are dropped from the map before the sandbox session is created
5. **Token pepper legacy fallback weakens hashed keys (CWE-916, medium):** `hash_token_candidates` returns only the peppered digest on peppered deployments; the legacy bare-SHA-256 digest is tried only when `LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1|true` is set explicitly, so rows written before the pepper was configured keep authenticating during migration and the fallback is opt-in, logged, and visible in config
6. **At-rest encryption falls back to the master key off-DB (CWE-321, medium):** `_get_salt_key` requires `LITELLM_SALT_KEY` unconditionally now; the previous fallback-to-master-key path (previously tolerated on DB-connected proxies) exists only behind `LITELLM_ALLOW_SALT_KEY_FALLBACK=1|true|True` and logs a warning when used, so at-rest encryption never silently degrades to a key derivable from the auth secret
7. **Proxy-Admin-Viewer GET default-allow (CWE-285, medium):** the safe-method branch no longer default-allows arbitrary GETs; it derives the read-only set from the live route table (paths whose registered methods contain no POST/PUT/PATCH/DELETE, snapshot invalidated on route-count change so lazily-mounted routers are picked up) and grants a viewer GET only on a genuinely read-only route, an explicitly allowlisted route, or a known global spend-tracking route; drifted write routes are additionally blocked verb-agnostically by suffix (`/service-account/request-rotation|approve|reject`, `/token`, `/register`) and the blocklist gained `/team/model/add`, `/team/model/delete`, `/model/block`, `/model/unblock`; `/claude-code/plugins` (a genuine GET+POST mixed route whose GET is a viewer read) is the one explicit addition to `admin_viewer_routes`
8. **SSO debug accepts expired session JWTs (CWE-613, medium):** `encode_ui_session_jwt` now injects `exp = now + LITELLM_UI_SESSION_DURATION` into every token that omits one (so a UI session JWT is always bounded server-side, matching the cookie Max-Age), and the SSO debug gate decodes with `verify_exp=True, require_exp=True`; the MCP OAuth cookie fallback and the onboarding decode also require `exp`
9. **Backend component allowlist unanchored startswith (CWE-708, medium):** the backend component's route predicate matched `path.startswith(prefix)`, so `/concurrent_request_logs_reset`-style neighbors of allowlisted prefixes passed; both components now share `component_route_match.path_matches_any`, which matches a prefix only at a segment boundary (exact path, or prefix + `/`), the backend list dropped its templated `/{mcp_server_name}/` catch-all in favor of the concrete `/authorize|token|register|mcp` templates plus the drift the old matching had been silently absorbing (`/bench/run/*`, `/service-account/*`, `/concurrent_request_logs`), and a coverage test now pins the predicate against the real route table
10. **Gateway component allowlist unanchored startswith (CWE-708, medium):** same segment-boundary fix via the shared matcher; the gateway list dropped its templated `/{provider}/` prefix in favor of the seven concrete route templates (`/{provider}/v1/batches`, `/{provider}/v1/files`, and their subpaths), and a drift test pins the predicate to the real route table so a new route template cannot silently fall outside the allowlist

#### Operator notes for commit 9

**Migration flag for legacy token rows.** On deployments that set `LITELLM_SALT_KEY` after keys were already hashed with bare SHA-256, set `LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1` until those rows have been re-hashed (a key is re-hashed on next rotation/login), then remove the flag; without it, pre-pepper keys stop authenticating immediately.

**At-rest encryption now requires LITELLM_SALT_KEY.** There is no automatic fallback to the master key anywhere; if the env var is unset, startup raises. `LITELLM_ALLOW_SALT_KEY_FALLBACK=1` restores the old behavior explicitly and logs a warning on use.

**Admin Viewer safe methods are default-deny now.** A viewer GET on a route that registers any write method is rejected unless the route is in `admin_viewer_routes` (this round added `/claude-code/plugins`) or the spend-tracking list. If a new read-only endpoint fails the derivation (mixed GET+POST route), add it to `admin_viewer_routes` in `litellm/proxy/_types.py`.

**UI session JWTs carry exp.** Tokens minted before this change have no `exp` claim and are rejected by the SSO debug gate, MCP OAuth cookie fallback, and onboarding decode; users just re-login.

### Commit 10: sixth re-scan remediation (echo-scan PR #1548 report)

All 10 findings are fixed, each with a regression test that fails if the fix regresses:

1. **Unbounded JSON schema in custom-code guardrail (CWE-674, low):** `json_schema_valid` pre-checks the schema with an iterative depth/size walk (`_MAX_JSON_SCHEMA_CHARS=1_000_000`, `_MAX_JSON_SCHEMA_DEPTH=50`) before handing it to the jsonschema library, so a self-referential `$ref` cycle or a pathological payload cannot drive unbounded recursion in either the walk or the validator
2. **Plugin registry clear+update mutation window (CWE-362, low):** `register_plugins` builds the new registry fully off to the side and rebinds it in a single assignment; readers never see a cleared or partially-updated registry (the clear+update sequence is gone)
3. **Concurrent-request-log sibling endpoints missing ownership checks (CWE-639, low):** `rate_limit_hits`, `operation_counts`, and `timeline` now run the same `_assert_user_can_view_token_rows` ownership gate as `ui_view_spend_logs`: admins pass, a user sees only their own key/alias, an unowned row is refused, and a team member sees team logs only through the existing team-membership check
4. **Raw exception echo in plugin proxy 502 and MCP registration (CWE-209, low):** the plugin-proxy 502 body is a fixed "Cannot connect to plugin" message (connection-error details stay in server logs only), and the MCP server registration failure handler returns the generic "Error registering mcp server" detail without the underlying exception text
5. **extract_all_files path normalization gap (CWE-22, medium):** skill file extraction validates every entry through the shared `is_safe_skill_file_path` predicate (rejecting absolute paths, `..` segments after normalization, backslashes, drive letters, and NUL bytes) in addition to the existing prefix strip, and requires the normalized path to equal the clean path, so a zip entry cannot escape the skill folder through any normalization difference; the predicate is public so both the extractor and its tests exercise the exact same rule
6. **Full spend-log rows written to debug logs (CWE-532, medium):** `ui_view_spend_logs` logs only counts (`rows_returned=N`, log level, file type, filename) at DEBUG; the rows themselves (request bodies, API keys, model params) never enter the debug stream
7. **user_url_validation bypass silently disabled SSRF (CWE-918, medium):** an unparseable URL in `user_url_validation` (or any site that resolves to a public/loopback/link-local target) no longer silently skips validation: it now warns once, loudly, per process, and the validated fetch path no longer follows redirects (each redirect target is validated the same way; a redirect to a disallowed host fails the fetch instead of bypassing the check)
8. **v2:gcm rows keep unsalted-SHA-256 KDF indefinitely (CWE-916, medium):** the encryption migration classifies `v2:gcm` rows as legacy, and `migrate_encryption` re-encrypts them to the HKDF-derived key on the same run, so the unsalted KDF cannot persist in the DB past one migration call
9. **Token digests fall back to bare SHA-256 silently (CWE-916, medium):** without `LITELLM_SALT_KEY`, `hash_token`/`hash_token_candidates` now raise (and proxy startup fails) unless `LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1|true` opts in to the legacy mode, which warns once per process; with the pepper set, lookups use only the peppered digest unless the same opt-in flag opens the legacy digest during a migration window (see commit 9 note 5)
10. **Admin Viewer auto-granted every GET-only route (CWE-863, medium):** viewer GET access is no longer derived from the route table; it is an explicit, fail-closed allowlist (`_ADMIN_VIEWER_ALLOWED_GET_ROUTES` in `route_checks.py` — health/status, analytics, listing/info, settings, guardrails/policy, and MCP reads, plus the masked `/credentials` reads) on top of the existing `admin_viewer_routes` and global spend-tracking lists; anything not classified is denied by default, and a drift test (`test_admin_viewer_get_route_classification_is_exhaustive`) fails whenever a new GET-only route reaches the viewer branch without being added to exactly one of the allow or deny set, so a future sensitive read endpoint cannot be silently granted

#### Operator notes for commit 10

**LITELLM_SALT_KEY is now required for DB-backed proxies.** Without it, `hash_token` raises and startup fails. To keep running unsalted (discouraged: digests are then deterministic global identifiers a DB leak confirms offline), set `LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1`, which logs a once-per-process warning.

**SSRF validation bypass is loud and redirect-safe.** Deployments that relied on an unparseable URL silently skipping `user_url_validation` now see one warning per process and a failed fetch. Redirects are no longer followed through to the final host; each hop must validate or the fetch fails.

**v2:gcm credential rows migrate in place.** Run the existing `migrate_encryption` endpoint once after deploying; it re-encrypts any remaining `v2:gcm` rows to the HKDF key.

**Admin Viewer GET access is an explicit allowlist.** A new GET-only endpoint is denied to viewers unless deliberately added to `_ADMIN_VIEWER_ALLOWED_GET_ROUTES` (or `admin_viewer_routes` / the global spend-tracking list). The drift test fails the build if a new GET route is left unclassified, so classification is a deliberate act, not a default.

### Commit 7: third re-scan remediation (echo-scan PR #1492 report)

All 12 findings are fixed, each with a regression test that fails if the fix regresses:

1. **Logo upload without content validation (CWE-434, low):** `/ui/upload/logo` checks the uploaded file's magic bytes against the accepted image signatures (PNG, JPG, GIF, WEBP, ICO) before writing anything to disk, enforces the existing 5MB size cap, and stores the file under `LITELLM_UPLOADS_DIR` (default `~/.litellm_uploads`) with a generated `logo_` prefix; a non-image payload (e.g. an SVG or a script renamed to `.png`) is rejected with a 400 before the write
2. **Unauthenticated SSO debug claim dump (CWE-200, low):** the `/sso/debug/login` and `/sso/debug/callback` endpoints are disabled unless `general_settings.enable_sso_debug` (or `ENABLE_SSO_DEBUG=true`) is set, and even then they require a proxy-admin UI session cookie; the rendered page allowlists the OIDC claims it prints, so tokens and internal identifiers never render
3. **Bearer token prefix strip via unanchored replace (CWE-20, low):** `get_bearer_token` strips exactly one leading `Bearer ` scheme (case-insensitive) from the start of the credential and rejects values with an embedded scheme token (`sk-one Bearer sk-two`) with a 400, so `sk-BearerSecret` style keys are no longer silently mangled
4. **Token hash without per-token salt (CWE-916, low):** `hash_token` (and `PrismaClient.hash_token`) now HMAC-SHA256 digests under `LITELLM_SALT_KEY` when it is set, so a database leak alone reveals nothing about the keys and digests differ across environments; when it is unset the output is byte-for-byte the legacy bare SHA-256; read paths (`hash_token_candidates`) try the peppered digest first and fall back to the legacy digest, so rows written before the pepper was configured keep authenticating with no migration
5. **Master key used directly as the UI session JWT secret (CWE-798, medium):** UI session cookies, BYOK session tokens, and onboarding session tokens are now signed with an HKDF-SHA256-derived key (`litellm-ui-session-jwt-v1` / `litellm-byok-session-jwt-v1` info strings) instead of the raw master key, so a leaked session cookie can never be used to verify or forge a master-key JWT (and vice versa); BYOK tokens additionally require an `exp` claim
6. **Skills sandbox executes model-generated code without a trust boundary (CWE-94, medium):** `requirements` and any bundled `requirements.txt` are validated before the sandbox session is created; `-e`/`--editable`, VCS URLs (`git+`, `svn+`, ...), `--index-url`/`--extra-index-url`, `--find-links`, `--trusted-host`, `--proxy`, and local filesystem paths are all rejected with a message that names only the offending token; validation errors never include host paths (the generic sandbox failure keeps returning `Sandbox execution failed`), and the container defaults to `network_mode: "none"` unless requirements are present (`LITELLM_SANDBOX_NETWORK_MODE` overrides)
7. **OAuth2 state validation relies solely on the cookie (CWE-352, medium):** the proxy mints a single-use `sso_state:{state}` cache entry at redirect time and consumes it at the callback; a state that was never issued by this proxy (or was already used) is rejected even when the attacker can set a matching cookie, and the entry is deleted on use so a replayed callback fails
8. **JWT audience/issuer verification silently disabled when env unset (CWE-347, medium):** `_build_decode_kwargs` raises unless `JWT_AUDIENCE` or `JWT_ISSUER` is configured (or `LITELLM_JWT_ALLOW_UNSCOPED=true` explicitly opts back into the legacy unscoped behavior, which warns once), so a JWT auth deployment without scoping fails closed instead of accepting any well-signed token
9. **OAuth2 introspection role unvalidated (CWE-863, medium):** the introspection response's role field is accepted only when it names a real `LitellmUserRoles` value, non-string roles are rejected, and tokens whose introspection response lacks `active: true` (or reports `active` missing) are refused; unauthenticated `proxy_admin` from a token-info path requires client credentials
10. **Proxy-Admin-Viewer default-allow keyed off HTTP method (CWE-285, medium):** the known write routes (`/key/delete`, `/key/update`, `/customer/update`, etc.) are blocked verb-agnostically before the safe-method default-allow, so a viewer key cannot reach a write route even via GET; suffix-parameter write routes (`/key/{key}/regenerate`, `/key/{key}/reset_spend`) are covered by the same mechanism
11. **Unauthenticated cross-tenant write `POST /customer/update` (CWE-639, medium):** `update_end_user` requires a proxy admin (`allow_view_only=False`), so viewer and all non-admin roles get a 403 before any DB write
12. **Unauthenticated `GET /customer/info` IDOR (CWE-639, medium):** `end_user_info` requires an admin session; viewer keys may read (`allow_view_only=True`) but every non-admin role gets a 403

#### Operator notes for commit 7

**Existing UI sessions invalidate on upgrade.** Session cookies, BYOK tokens, and onboarding tokens minted before this change were signed with the raw master key; the new code verifies with the HKDF-derived key, so users are simply logged out and re-login fixes it. No data migration is involved

**Token hashes change when `LITELLM_SALT_KEY` is set.** New writes store the peppered HMAC digest, while lookups accept both the peppered and the legacy bare-SHA-256 digests, so existing keys keep working. Set `LITELLM_SALT_KEY` (already used for at-rest encryption) before the first keys are minted on a new deployment; leaving it unset preserves the exact legacy digests

**JWT auth deployments need scoping configured.** A proxy relying on JWT auth with neither `JWT_AUDIENCE` nor `JWT_ISSUER` set now refuses requests with a 401 explaining the requirement; deployments that genuinely accept unscoped tokens can set `LITELLM_JWT_ALLOW_UNSCOPED=true` to restore the old behavior (a warning is logged once per process)

**SSO debug endpoints are opt-in.** `/sso/debug/login` and `/sso/debug/callback` return 404 unless `enable_sso_debug` is set in `general_settings` (or `ENABLE_SSO_DEBUG=true`), and then still require a proxy-admin session cookie; the pages print only allowlisted OIDC claims

**Uploads and sandbox defaults changed.** Logo uploads land in `LITELLM_UPLOADS_DIR` (default `~/.litellm_uploads`) with magic-byte validation; a previously accepted SVG upload is now rejected. Skills sandboxes run with no network unless the skill carries requirements, and `LITELLM_SANDBOX_NETWORK_MODE` overrides for operators who need the old behavior

**Customer endpoints are admin-gated.** Both `/customer/info` and `/customer/update` now require a proxy-admin session key (viewer keys may still read `/customer/info`); scripts or dashboards that called these endpoints with weaker keys need to switch to an admin key

### Commit 6: second re-scan remediation (echo-scan PR #1048 report)

All 11 findings are fixed, each with a regression test that fails if the fix regresses:

1. **Guardrail HTTP SSRF default-allow:** the `CUSTOM_CODE_HTTP_ALLOWED_HOSTS` allowlist now denies by default when unset, and every guardrail HTTP primitive additionally routes through the shared SSRF validator, so loopback, RFC1918, link-local, and cloud-metadata destinations are refused even with the allowlist off.
2. **Unverified SSO access-token claims:** `process_sso_jwt_access_token` verifies the provider access token against the JWKS (`_verify_sso_access_token`) before any team/role extraction; verification failure fails closed (no claims, no role mapping), so an attacker-crafted RS256/HS256 payload can no longer drive role assignment.
3. **Verbose DB/client error passthrough:** error responses are truncated and redacted so raw provider/database error bodies no longer reach the caller.
4. **Internally computed identifiers interpolated into SQL:** the user-agent analytics endpoints bind all internally computed identifiers as `$N` parameters instead of f-string interpolation.
5. **MCP server URL trusted at call time:** stdio/SSE/HTTP MCP server destinations are re-validated at call time, not just at config load.
6. **At-rest encryption master-key fallback:** `_get_salt_key` refuses the `LITELLM_MASTER_KEY` fallback on a DB-connected proxy (`prisma_client is not None`); AES-256-GCM + HKDF is the default for new writes; a new `v4:ctx:` format binds each ciphertext to its `table:field` context via AES-GCM AAD so a value lifted from one column fails authentication in another; every encrypt/decrypt pair in the proxy (config env vars, SSO settings, credentials, model litellm_params, MCP tables, vantage/cloudzero, callback vars, rotation and migration walkers) threads the context through, and the migration engine recognizes all three markers.
7. **Plugin session claim role staleness:** `verify_plugin_session_claim` re-resolves the caller's live role from the user cache/DB on every verification; a demoted or deactivated user fails verification immediately instead of riding the 30s claim TTL, and the claim's embedded role is treated as a hint only.
8. **MCP stdio spawn TOCTOU:** `_resolve_mcp_stdio_command` returns a pinned absolute path (realpath'd, opened with `O_NOFOLLOW`, fstat-verified as a regular file, not group/world-writable, owned by root or the proxy euid, with every ancestor directory checked for world-writability), and `_create_mcp_client` hands that exact path to the spawn config, so PATH is never consulted at exec time and the checked object is the executed object.
9. **Self-service user update field escalation:** `_check_user_update_authz` restricts non-admin self-updates to a positive allowlist (`user_alias`, `user_email`, `password`, `metadata`) via `model_fields_set` presence checks; `teams`, `models`, `tpm/rpm limits`, `budget_duration`, `guardrails`, `policies`, and every other privileged field require a proxy admin.
10. **Unbounded months/date values in SQL:** the WAU/MAU analytics queries bind `months`, `end_date`, and the week-bucket offsets as positional parameters; f-strings are limited to fixed identifiers.
11. **Plugin proxy credential/header hardening:** the shared plugin session claim carries no litellm bearer token, the plugin_key is injected server-side only, and all litellm credential headers are stripped before forwarding.

#### Operator notes for commit 6

**Salt key is now required on DB-connected proxies.** `_get_salt_key` raises instead of falling back to `LITELLM_MASTER_KEY` once `prisma_client` is connected, because the master key is also a bearer credential and using it as the at-rest encryption key means rotating it silently strands every stored provider key, OAuth token, and MCP credential. Set `LITELLM_SALT_KEY` before upgrading a deployment that connects to a database. Deployments that genuinely cannot set it yet can set `LITELLM_ALLOW_SALT_KEY_FALLBACK=1` to explicitly opt back into the legacy fallback for a migration window; the proxy logs the refusal with a docs link when it fires. SDK-only use (no proxy import) is unaffected.

**New at-rest ciphertext formats are additive.** Decrypt is format-detecting, so all historical values keep reading: legacy XSalsa20-Poly1305, `v2:gcm:` (SHA-256 key), `v3:hkdf:` (HKDF-SHA256), and now `v4:ctx:<len>:<context>...` where the AES-GCM AAD binds the value to its `table:field` storage context. A `v4` value read with the wrong or missing context fails closed, so a ciphertext lifted from one column can never be replayed into another. Rotation (`rotate_master_key`) and the migration engine re-encrypt through the same context-aware helpers, so no data migration step is required on upgrade; new writes simply use the stronger format. Deployments that want the legacy algorithm for new writes can set `general_settings.encryption_algorithm: xsalsa20-poly1305` explicitly; the default for new deployments is `aes-256-gcm`.

**Guardrail HTTP egress is deny-by-default.** Custom-code guardrails that use the injected `http_request`/`http_get`/`http_post` primitives now need `CUSTOM_CODE_HTTP_ALLOWED_HOSTS` set to make outbound calls at all (comma-separated exact hostnames). With the allowlist set, destinations additionally pass through the shared SSRF validator, so loopback, RFC1918, link-local, and cloud-metadata addresses are refused regardless. A guardrail that previously reached an internal host will start failing its HTTP calls until the host is allowlisted.

**MCP stdio commands resolve under stricter rules.** The allowlist still applies (`LITELLM_MCP_STDIO_EXTRA_COMMANDS` extends it), but the spawn now pins the realpath of the binary and refuses it if the file itself is group- or world-writable, is not owned by root or the proxy's own user, or sits under any world-writable ancestor directory. A binary that previously spawned from a shared or mutable path may now be rejected with a 403 naming the exact condition; move it to a protected location (or fix its mode/ownership) rather than loosening the check.

**Non-admin self-updates are allowlisted.** A non-admin calling `/user/update` on their own row may now change only `user_alias`, `user_email`, `password`, and `metadata`. Requests that also carry `models`, `tpm_limit`, `rpm_limit`, `max_parallel_requests`, `budget_duration`, `guardrails`, `policies`, `blocked`, `aliases`, or any other privileged field are rejected with a 403 that names the offending fields, even when the intent was an innocent profile edit. Dashboard flows that let users edit their own profile should send only the permitted fields; anything else goes through a proxy admin. `teams` and `organization_id` are not fields on `UpdateUserRequest` and were never persisted through this endpoint.

**SSO deployments using access-token claims need `GENERIC_JWKS_ENDPOINT`.** Where the proxy previously read `groups`/role attributes from the provider's access token without signature verification, it now verifies the token against the JWKS first and extracts no claims from a token that fails verification (the login still proceeds; only JWT-derived team/role assignment is skipped). Deployments relying on access-token-driven role mappings must set `GENERIC_JWKS_ENDPOINT` (a JWKS URL or an OIDC discovery URL) or those mappings silently stop applying, with an error logged per login.

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

### echo-scan second re-scan (PR #1048 report)

The second re-scan still found 11 findings open (0 critical / 2 high / 6 medium / 3 low), all deeper variants of the earlier clusters: default-allow egress in the guardrail HTTP primitives, unverified access-token claims, the master-key fallback for at-rest encryption, claim TTL staleness in the plugin session, a TOCTOU window in the MCP stdio spawn, and a field allowlist gap in non-admin self-updates. Commit 6 clears all of them with the fixes itemized above.

### echo-scan third re-scan (PR #1492 report)

The third re-scan still found 12 findings open (0 critical / 0 high / 8 medium / 4 low), all deeper variants of the earlier clusters: upload content validation, session-cookie trust boundaries, OAuth2 state and introspection hardening, JWT scope fail-closed, viewer write-route gating, and customer-endpoint authorization. Commit 7 clears all of them with the fixes itemized above.

### echo-scan fourth re-scan (PR #1511 report)

The fourth re-scan found 8 findings open (0 critical / 1 high / 4 medium / 3 low). Three of them sat on surface the earlier fixes themselves introduced (the sandbox requirements validator, the viewer blocklist, and the session-cookie helper); the rest were deeper variants of known clusters. Commit 8 clears all of them: the sandbox requirements validator now parses every line with packaging's PEP 508 `Requirement` parser and rejects any line that names a URL or fails to parse, so pip options, `[options]` sections, per-requirement flags, and `pkg @ url` forms are all refused structurally instead of by regex; the UI path-rewrite walk follows symlinks only outward (realpath containment against the UI root, plus per-file OSError/PermissionError logging) and lives in a testable module-level function; the proxy-admin-viewer write blocklist is a single authoritative module-level frozenset (the weaker class-level copy is gone, so `/team/block`, `/model/update`, and the `/key/{id}/regenerate|reset_spend` suffixes can no longer drift open); JSON login responses now set the session cookie HttpOnly by default while the SSO RedirectResponse stays JS-readable (the SSO callback delivers the token only via the cookie, so HttpOnly there would break login; `LITELLM_UI_SESSION_COOKIE_HTTPONLY` forces both shapes); the MCP discovery module's three outbound fetches (well-known GET, token exchange POST, DCR registration POST) all route through the SSRF-validating `safe_get`/`async_safe_post` helpers, with POSTs refusing to follow redirects; the SSO id_token signature algorithm is checked against the asymmetric allowlist (`JWTHandler.SUPPORTED_JWT_ALGORITHMS`) instead of being taken from the attacker-controlled JOSE header, and the JWKS key is selected uniquely by `kid`; and the two remaining auth-path `assert`s (team-object fallback, non-admin key self-creation) are now explicit 403 exceptions that survive `python -O`.

Operator notes: requirements are PEP 508 now, so extras and environment markers are accepted while every form of index/URL redirection is rejected; if a provider's MCP metadata legitimately points at an internal host, allow it explicitly via `user_url_allowed_hosts` in general settings (the same knob as the other SSRF-validated fetches); providers that sign id_tokens with HS256 against a JWKS will now be rejected by design (use an asymmetric provider).

### echo-scan fifth re-scan (PR #1527 report)

The fifth re-scan found 10 findings open (0 critical / 0 high / 7 medium / 3 low). Two were the same unanchored-prefix bug class in the backend and gateway component allowlists (introduced by the component split), and the rest were hardening gaps in surface the earlier rounds touched or deferred. Commit 9 clears all of them: the custom-code guardrail's regex and JSON primitives are size-capped before they touch the engine or parser; plugin session claims are proxy-issued (a proxy-side issuer MAC the plugin's Fernet key cannot derive, checked before decryption) so a plugin can read claims but never mint them; logo uploads are re-encoded through Pillow to PNG so polyglot bytes never reach disk, and served uploaded assets are `Content-Disposition: attachment`; sandbox skill files are `PurePosixPath`-validated (no absolute, `..`, backslash, or NUL paths); the token pepper's legacy bare-SHA-256 fallback is behind `LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`; at-rest encryption requires `LITELLM_SALT_KEY` with the master-key fallback only behind `LITELLM_ALLOW_SALT_KEY_FALLBACK`; the Admin Viewer safe-method default-allow is replaced by a live route-table derivation (GET-only registered paths) with drift-proof suffix blocks; UI session JWTs always carry `exp` and the SSO debug gate, MCP OAuth cookie fallback, and onboarding decode verify it; and both component allowlists match prefixes only at segment boundaries via a shared matcher, with templated catch-alls replaced by concrete route-template enumerations pinned by coverage tests.

### echo-scan sixth re-scan (PR #1548 report)

The sixth re-scan found 10 findings open (0 critical / 0 high / 6 medium / 4 low). The largest was the Admin Viewer's route-table-derived GET access (the very mechanism commit 9 introduced), which the scanner correctly flagged as a default-allow: it is now an explicit fail-closed allowlist with a classification-drift test forcing every new GET-only route to be classified deliberately. The rest were residual gaps on surface earlier rounds created or only partially hardened: the custom-code JSON schema primitive now pre-checks depth/size before the jsonschema library sees it; the plugin registry update is an atomic rebind; the concurrent-request-log siblings gained the ownership gate their sibling already had; plugin-proxy 502s and MCP registration errors stopped echoing raw exception text; skill-file extraction validates through a shared path-safety predicate; spend-log debug output is counts-only; the SSRF validation bypass is loud and redirect-proof instead of silent; and both the v2:gcm encryption rows and unsalted token digests now fail closed instead of silently persisting weak primitives. Commit 10 clears all of them.

## Pre-Submission checklist

- [x] Meaningful tests added (token_counter download-cap tests, url_utils scheme tests, auth_utils allowlist tests, ui_sso alg tests, azure/anthropic count_tokens destination-guard tests, provider_url_destination_guard tests, plugin-proxy traversal and salt fail-closed tests, MCP stdio validation tests, session-cookie flag tests, host-trust redirect tests, tag-filter parameter-binding tests, ORDER BY allowlist tests, JWKS verification tests including forged-signature and wrong-audience fail-closed cases, HKDF derivation tests, custom-code allowlist and timeout tests, salt-key fallback guard tests, context-bound ciphertext transplant tests, plugin claim live-role re-resolution tests, stdio pinned-path and world-writable-file/ancestor tests, self-update privileged-field allowlist tests, logo magic-byte and role-gate tests, SSO debug gate and claim-allowlist tests, bearer-scheme parse tests, peppered token-hash and legacy-digest fallback tests, sandbox requirements-validation and network-mode tests, OAuth2 state single-use tests, JWT scope fail-closed tests, introspection role and active-flag tests, viewer write-route block tests, customer-endpoint admin-gate tests, guardrail primitive size-cap and oversized-refusal tests, plugin-claim forge and tamper tests, logo Pillow re-encode and polyglot-stripping plus attachment-header tests, sandbox unsafe-skill-path drop tests, viewer route-table-derivation drift and unknown-GET denial tests, encode-exp injection and require-exp tests, segment-boundary allowlist and templated-entry absence tests, json-schema depth/size refusal and json_parse cap tests, plugin registry atomic-rebind test, concurrent-request-log sibling ownership tests, plugin-proxy 502 and MCP registration exception-echo suppression tests, extract_all_files traversal/windows-path drop tests and shared path-safety predicate tests, spend-log debug counts-only test, SSRF bypass loud-warning and no-redirect-follow tests, v2:gcm migration-to-HKDF test, unsalted token-digest fail-closed and legacy opt-in warn-once tests, viewer explicit-allowlist classification-drift and unlisted-route denial tests)
- [ ] CI/CD checks pass
- [x] Scope isolated: ten commits, one per audit plus one lint cleanup
- [ ] Greptile review requested

## Type

🐛 Bug Fix
🚄 Infrastructure
✅ Test
