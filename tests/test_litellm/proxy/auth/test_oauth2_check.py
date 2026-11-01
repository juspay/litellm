import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.oauth2_check import Oauth2Handler


@pytest.fixture
def _oauth_env(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setenv("OAUTH_TOKEN_INFO_ENDPOINT", "https://idp.example.com/introspect")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.delenv("OAUTH_USER_ID_FIELD_NAME", raising=False)
    monkeypatch.delenv("OAUTH_USER_ROLE_FIELD_NAME", raising=False)
    monkeypatch.delenv("OAUTH_USER_TEAM_ID_FIELD_NAME", raising=False)


def _mock_http_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    return response


def _introspection_payload(**overrides) -> dict:
    payload = {
        "active": True,
        "sub": "user-1",
        "role": "internal_user",
        "team_id": "team-1",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_check_oauth2_token_valid_role_round_trip(_oauth_env):
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload()))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        result = await Oauth2Handler.check_oauth2_token(token="tok-1")

    assert result.user_id == "user-1"
    assert result.user_role == LitellmUserRoles.INTERNAL_USER
    assert result.team_id == "team-1"


@pytest.mark.asyncio
async def test_check_oauth2_token_rejects_inactive_token(_oauth_env):
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload(active=False)))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="Token is not active"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_rejects_missing_active_field(_oauth_env):
    payload = _introspection_payload()
    del payload["active"]
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(payload))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="Token is not active"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_rejects_unknown_role(_oauth_env):
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload(role="superadmin")))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="invalid role"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_rejects_non_string_role(_oauth_env):
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload(role=1337)))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="invalid role"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_refuses_proxy_admin_without_client_credentials(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setenv("OAUTH_TOKEN_INFO_ENDPOINT", "https://idp.example.com/introspect")
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_SECRET", raising=False)

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload(role="proxy_admin")))
    mock_client.get = AsyncMock(return_value=_mock_http_response(_introspection_payload(role="proxy_admin")))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="client credentials"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_allows_proxy_admin_with_client_credentials(_oauth_env):
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=_mock_http_response(_introspection_payload(role="proxy_admin")))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        result = await Oauth2Handler.check_oauth2_token(token="tok-1")

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
async def test_check_oauth2_token_info_get_path_enforces_active(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setenv("OAUTH_TOKEN_INFO_ENDPOINT", "https://idp.example.com/tokeninfo")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=_mock_http_response(_introspection_payload(active=False)))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="Token is not active"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")


@pytest.mark.asyncio
async def test_check_oauth2_token_validates_role_on_get_path(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setenv("OAUTH_TOKEN_INFO_ENDPOINT", "https://idp.example.com/tokeninfo")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=_mock_http_response(_introspection_payload(role="made-up-role")))

    with patch("litellm.proxy.auth.oauth2_check.get_async_httpx_client", return_value=mock_client):
        with pytest.raises(ValueError, match="invalid role"):
            await Oauth2Handler.check_oauth2_token(token="tok-1")
