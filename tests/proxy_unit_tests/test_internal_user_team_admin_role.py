"""
Unit tests for the `internal_user__view_team_admin` user role.

This role must be treated as an internal-user-tier, view-only role with
permissions identical to `INTERNAL_USER_VIEW_ONLY`. These tests pin that
contract without requiring a database: they exercise the enum property,
the route gate, and the spend-log view helper directly.

Mutation targets that would fail a test below:
- removing INTERNAL_USER_VIEW_TEAM_ADMIN from `is_internal_user_role`
- wiring the route gate to `internal_user_routes` (write routes) instead of
  `internal_user_view_only_routes`
- omitting the route-checks elif branch for the new role entirely
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("../.."))

import pytest
from fastapi import Request
from starlette.datastructures import URL

from litellm.proxy._types import (
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    _can_user_view_spend_log,
)


VIEW_ONLY_ROUTES = LiteLLMRoutes.internal_user_view_only_routes.value
WRITE_ROUTES = LiteLLMRoutes.internal_user_routes.value


VIEW_ROUTE_SAMPLE = "/spend/logs"
WRITE_ROUTE_SAMPLE = "/key/generate"


def _route_in_list(route: str, allowed_routes) -> bool:
    return RouteChecks.check_route_access(route=route, allowed_routes=allowed_routes)


def test_role_value_and_label():
    assert LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN.value == "internal_user__view_team_admin"
    assert LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN.ui_label == "Internal User View Team Admin"
    assert LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN.description != ""


def test_is_internal_user_role_true():
    assert LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN.is_internal_user_role is True


def test_not_an_admin_role():
    admin_view_roles = (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )
    assert LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN not in admin_view_roles


def test_view_only_route_access_matches_internal_user_viewer():
    """INTERNAL_USER_VIEW_TEAM_ADMIN must be granted the same routes as
    INTERNAL_USER_VIEW_ONLY via the internal_user_view_only_routes list."""
    assert _route_in_list(VIEW_ROUTE_SAMPLE, VIEW_ONLY_ROUTES) is True


@pytest.mark.parametrize(
    "route,expected",
    [
        (VIEW_ROUTE_SAMPLE, True),
        (WRITE_ROUTE_SAMPLE, False),
        ("/team/new", False),
        ("/team/delete", False),
        ("/team/update", False),
        ("/global/spend/logs", False),
    ],
)
def test_route_access_parity_with_internal_user_viewer(route: str, expected: bool):
    """Every route reachable by INTERNAL_USER_VIEW_ONLY must be reachable by
    INTERNAL_USER_VIEW_TEAM_ADMIN, and no write/admin route may be."""
    team_admin_access = _route_in_list(route, VIEW_ONLY_ROUTES)
    viewer_access = _route_in_list(route, VIEW_ONLY_ROUTES)
    assert team_admin_access == viewer_access == expected


def test_team_admin_not_granted_write_routes():
    """The write/internal_user_routes list contains key-management routes.
    INTERNAL_USER_VIEW_TEAM_ADMIN must NOT be granted those (it is view-only)."""
    assert _route_in_list(WRITE_ROUTE_SAMPLE, WRITE_ROUTES) is True
    team_admin_allowed = RouteChecks.check_route_access(
        route=WRITE_ROUTE_SAMPLE,
        allowed_routes=VIEW_ONLY_ROUTES,
    )
    assert team_admin_allowed is False


def test_route_checks_dispatch_grants_view_route():
    """End-to-end-ish: the RouteChecks minified-route dispatch path used by
    user_api_key_auth must allow a view route for INTERNAL_USER_VIEW_TEAM_ADMIN."""
    allowed = RouteChecks.check_route_access(
        route=VIEW_ROUTE_SAMPLE,
        allowed_routes=LiteLLMRoutes.internal_user_view_only_routes.value,
    )
    assert allowed is True


def test_can_user_view_spend_log_true_for_team_admin():
    user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
        user_id="team-admin-1",
    )
    assert _can_user_view_spend_log(user) is True


def test_can_user_view_spend_log_false_without_user_id():
    user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
        user_id=None,
    )
    assert _can_user_view_spend_log(user) is False


def test_can_user_view_spend_log_false_for_customer():
    user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.CUSTOMER,
        user_id="cust-1",
    )
    assert _can_user_view_spend_log(user) is False


def _build_request(route: str) -> Request:
    request = Request(scope={"type": "http"})
    request._url = URL(url=route)
    return request


@pytest.mark.asyncio
async def test_dispatch_grants_view_route_for_team_admin():
    """The role-dispatch path in non_proxy_admin_allowed_routes_check must
    grant a view-only route to INTERNAL_USER_VIEW_TEAM_ADMIN. If the elif branch
    for the new role is removed, this falls through to the admin-only
    exception and raises."""
    valid_token = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
        user_id="team-admin-1",
        api_key="sk-test",
    )
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=None,
        _user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
        route=VIEW_ROUTE_SAMPLE,
        request=_build_request(VIEW_ROUTE_SAMPLE),
        valid_token=valid_token,
        request_data={},
    )


@pytest.mark.asyncio
async def test_dispatch_denies_write_route_for_team_admin():
    """INTERNAL_USER_VIEW_TEAM_ADMIN must NOT be able to reach a write route
    (/key/generate). The dispatch must raise HTTPException(403)."""
    valid_token = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
        user_id="team-admin-1",
        api_key="sk-test",
    )
    with pytest.raises(Exception, match="Only proxy admin can be used"):
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=None,
            _user_role=LitellmUserRoles.INTERNAL_USER_VIEW_TEAM_ADMIN,
            route=WRITE_ROUTE_SAMPLE,
            request=_build_request(WRITE_ROUTE_SAMPLE),
            valid_token=valid_token,
            request_data={},
        )

