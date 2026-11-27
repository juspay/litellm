# PR Summary: New user role `internal_user__view_team_admin`

**Branch:** `litellm_internal_user_team_admin` (branched from `release/v1.92.0`)

## Relevant issues

N/A

## Linear ticket

Resolves

## Summary

This PR introduces a new global user role, `internal_user__view_team_admin` (UI label: "Internal User View Team Admin"), that carries exactly the same permissions as `internal_user_viewer` (Internal User — View Only). A user assigned this role can log in, view their own keys, and view their own spend, but has no write access of any kind: no key generation, update, deletion, or regeneration, no team create/update/delete, and no access to proxy-wide spend.

The role is opt-in only. It is assigned explicitly via user creation, user update, bulk update, SCIM provisioning, organization membership, or SSO group role mapping. No default role anywhere in the codebase changed; new users without a specified role still default to `internal_user_viewer`.

The role value deliberately uses the `internal_user` prefix to signal that it belongs to the internal-user tier, and avoids the bare `team_admin` string, which would collide conceptually with the existing per-team membership role `LiteLLMTeamRoles.TEAM_ADMIN` (a separate enum whose value is `"admin"`).

## Key design decision: one centralized tier check

The pre-existing code granted internal-user-tier permissions through scattered explicit comparisons of the form `role == INTERNAL_USER or role == INTERNAL_USER_VIEW_ONLY`, repeated across roughly ten sites in spend tracking. Adding the new role by appending a third `or` clause to each site would have been mechanical but fragile: the next internal-tier role would require another ten edits, and a missed site would silently produce a permission gap.

Instead, this PR routes all of those sites through the existing `is_internal_user_role` property on the `LitellmUserRoles` enum, which now covers `internal_user`, `internal_user_viewer`, and `internal_user__view_team_admin`. Future internal-tier roles become a one-line enum addition plus one route-gate entry.

## Changes

### Backend

**`litellm/proxy/_types.py`**
- New enum member `LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN = "internal_user__view_team_admin"`, with description and UI label entries
- `is_internal_user_role` now includes the new role, making every internal-tier check cover it
- New role added to the `user_role` Literal unions on `NewUserRequest`, `NewUserResponse`, and `UpdateUserRequestNoUserIDorEmail`, so it is a valid assignable role in API requests and typed responses
- New role added to `OrgMember.role` and `ROLES_WITHIN_ORG`, so org admins can assign it within their organization
- `user_role_doc_str` updated to document the new value

**`litellm/proxy/auth/route_checks.py`**
- New gate branch in `non_proxy_admin_allowed_routes_check` that authorizes `internal_user__view_team_admin` against `internal_user_view_only_routes`, the exact same route list used by `internal_user_viewer`. Write and key-management routes stay denied.

**`litellm/proxy/spend_tracking/spend_management_endpoints.py`**
- Eight explicit two-role comparisons replaced with the centralized `is_internal_user_role` check. Behavior for existing roles is unchanged; the new role inherits internal-user spend scoping (own-spend filtering, own-key scoping) through the single property
- `_can_user_view_spend_log` now uses the same property

**`litellm/proxy/rag_endpoints/endpoints.py`**
- The existing-vector-store-only restriction on RAG ingest (view-only roles may ingest only into an existing vector store, never create new ones) now also applies to the new role, matching `internal_user_viewer` exactly

**`litellm/proxy/management_endpoints/ui_sso.py`**
- New role appended to the SSO role hierarchy used by `determine_role_from_groups`, so SSO group-to-role mappings can grant it

**`litellm/proxy/management_endpoints/scim/scim_v2.py`**
- New role added to the `ScimUserRole` assignable-role Literal. The SCIM default role remains `internal_user_viewer`

**`litellm/proxy/management_endpoints/internal_user_endpoints.py`**
- `/user/available_roles` now returns the new role with its description and UI label, so the Create User UI dropdown picks it up automatically

**`litellm/proxy/auth/login_utils.py`**
- Local `user_role` type annotation extended for type completeness. No behavior change; the invite-link fallback default remains `internal_user_viewer`

### Dashboard

- `src/utils/roles.ts`: the role is included in `internalUserRoles` (sidebar and permission gating) and `formatUserRole` renders "Internal User View Team Admin"
- `src/components/user_dashboard.tsx`: role display name mapping
- `src/components/bulk_create_users_button.tsx`: role accepted in CSV bulk-create validation and documented in the template help text
- `src/components/organization/organization_view.tsx`: role added to the add-member and edit-member role options
- SSO role mapping forms (`SSOModals.tsx`, `Settings/AdminSettings/SSOSettings/*`): the role is a selectable Default Role, with a new "Internal User View Team Admin Teams" group input; the payload builder sends `internal_user__view_team_admin` groups through the same path as the other roles
- `src/lib/http/schema.d.ts`: regenerated from the proxy OpenAPI spec via `node scripts/gen-api-types.mjs`

### Tests

**`tests/proxy_unit_tests/test_internal_user_team_admin_role.py`** (new, 17 tests)
- Enum contract: value, UI label, description, `is_internal_user_role` is True, and it is not an admin role
- Route parity: for the route lists, the role resolves to the same allow/deny decisions as `internal_user_viewer`, including denial of `/key/generate`, `/team/new`, `/team/update`, and `/global/spend/logs`
- Full dispatch: `non_proxy_admin_allowed_routes_check` grants a view route (`/spend/logs`) and raises on a write route (`/key/generate`) for a caller with the new role
- Spend-log helper: `_can_user_view_spend_log` returns True for the new role with a user_id, False without one, and False for `customer`
- Mutation-verified: removing the route-gate branch causes `test_dispatch_grants_view_route_for_team_admin` to fail; removing the role from `is_internal_user_role` causes the property test and the spend-log test to fail

**`tests/test_litellm/proxy/auth/test_route_checks.py`**
- New role added to the `test_available_roles_accessible_to_non_admin_users` parametrization, confirming `/user/available_roles` passes the route check for it

**`tests/proxy_admin_ui_tests/test_role_based_access.py`**
- The DB-backed `test_user_role_permissions` parametrization extended with parity cases for the new role (denied writes, allowed `/spend/logs`)

**Dashboard tests updated for the new SSO mapping field** (`SSOModals.test.tsx`, `SSOSettings/utils.test.ts`, `EditSSOSettingsModal.test.tsx`): expected payloads and form values now include `internal_user__view_team_admin` entries, including a whitespace-trim case for the new team mapping input

## What is deliberately NOT included

- The new role is not added to `internal_user_routes` (the create/delete internal-user route list) — it is view-only
- No default role changed: SCIM default, SSO fallback, invite-link fallback, and public-route fallback all remain `internal_user_viewer`
- `LiteLLMTeamRoles.TEAM_ADMIN` (per-team membership role, value `"admin"`) is untouched; the new global role does not interact with it

## Pre-Submission checklist

- [x] I have added meaningful tests
- [x] My PR passes all CI/CD checks (e.g., lint, format, unit tests)
- [x] My PR's scope is as isolated as possible; it only solves 1 specific problem
- [ ] I have requested a Greptile review by commenting `@greptileai` and received a **Confidence Score of at least 4/5** before requesting a maintainer review

## Type

🆕 New Feature

## Screenshots / Proof of Fix

Ruff format and lint are clean, and the full affected suites pass: 408/408 backend tests across the new role test file and `test_route_checks.py`, 298/298 dashboard tests for SSO settings, SSOModals, and role utilities. Pre-existing failures unrelated to this change (missing provider config in local env, DB-dependent tests) were confirmed present on the base branch via `git stash` comparison.

End-user verification against a live proxy (localhost:4000, master key `sk-1234`):

1. Start the proxy:
   ```bash
   python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
   ```
2. Confirm the role is available to the UI:
   ```bash
   curl -s http://localhost:4000/user/available_roles -H "Authorization: Bearer sk-1234"
   ```
   Expected: the response includes `"internal_user__view_team_admin"` with UI label "Internal User View Team Admin"
3. Create a user with the new role:
   ```bash
   curl -s -X POST http://localhost:4000/user/new -H "Authorization: Bearer sk-1234" -H "Content-Type: application/json" \
     -d '{"user_email": "ta-test@example.com", "user_role": "internal_user__view_team_admin"}'
   ```
4. Generate a key for that user (as admin), then with that key confirm view-only parity:
   ```bash
   curl -s http://localhost:4000/spend/logs -H "Authorization: Bearer <ta-test-key>"        # 200, own spend only
   curl -s -X POST http://localhost:4000/key/generate -H "Authorization: Bearer <ta-test-key>" -H "Content-Type: application/json" -d '{}'   # 403
   curl -s -X POST http://localhost:4000/team/new -H "Authorization: Bearer <ta-test-key>" -H "Content-Type: application/json" -d '{"team_alias": "x"}'   # 403
   ```
5. UI: open http://localhost:4000/ui, go to Users, click "Invite User", and confirm the User Role dropdown shows "Internal User View Team Admin"
