"""
Unit tests for the `internal_user__view_team_admin` user role.

This role must be treated as an internal-user-tier, view-only role with
permissions identical to `INTERNAL_USER_VIEW_ONLY`. These tests pin that
contract without requiring a database: they exercise the enum property,
the route gate, and the spend-log view helper directly.

Every test below fails when a specific part of the feature is broken:
- test_is_internal_user_role_true: remove the role from is_internal_user_role
- test_can_user_view_spend_log_true_for_team_admin: same removal
- test_dispatch_grants_view_route_for_team_admin: remove the role from the
  route_checks elif, or wire it to internal_user_routes (write routes)
- test_dispatch_denies_write_route_for_team_admin: wire the role to
  internal_user_routes instead of internal_user_view_only_routes
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
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    _can_user_view_spend_log,
)


VIEW_ROUTE_SAMPLE = "/spend/logs"
WRITE_ROUTE_SAMPLE = "/key/generate"


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
    grant a view-only route to INTERNAL_USER_VIEW_TEAM_ADMIN. If the role is
    removed from the elif, this falls through to the admin-only exception."""
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
    (/key/generate). If the role is wired to internal_user_routes instead of
    internal_user_view_only_routes, this stops raising and the test fails."""
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
