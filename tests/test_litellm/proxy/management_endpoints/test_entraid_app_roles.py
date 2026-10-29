import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.management_endpoints.ui_sso import MicrosoftSSOHandler
from litellm.proxy.management_endpoints.types import get_litellm_user_role
from litellm.proxy._types import LitellmUserRoles

_TENANT = "tenant-id"
_CLIENT_ID = "litellm-app"
_ISSUER = f"https://login.microsoftonline.com/{_TENANT}/v2.0"


def _jwks_for_key(public_key, kid: str) -> dict:
    import base64

    pub = public_key.public_numbers()

    def b64uint(n: int) -> str:
        b = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(b).decode().rstrip("=")

    return {
        "keys": [
            {"kty": "RSA", "kid": kid, "n": b64uint(pub.n), "e": b64uint(pub.e), "alg": "RS256"}
        ]
    }


class _FakeClient:
    def __init__(self, jwks: dict):
        self._jwks = jwks

    async def get(self, url: str):
        class _Resp:
            status_code = 200

            def json(self_inner):
                return self._jwks

        return _Resp()


@pytest.fixture
def entra_env(monkeypatch):
    monkeypatch.setenv("MICROSOFT_TENANT", _TENANT)
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", _CLIENT_ID)
    monkeypatch.setenv("GENERIC_JWKS_ENDPOINT", f"https://login.microsoftonline.com/{_TENANT}/discovery/v2.0/keys")


def _patch_jwks(monkeypatch, public_key, kid: str):
    from litellm.proxy.management_endpoints import ui_sso

    monkeypatch.setattr(
        ui_sso,
        "get_async_httpx_client",
        lambda llm_provider, params=None: _FakeClient(_jwks_for_key(public_key, kid)),
    )


def _mint(private_key, payload, kid: str) -> str:
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


def _payload(app_roles=None) -> dict:
    payload = {
        "sub": "user123",
        "email": "admin@company.com",
        "aud": _CLIENT_ID,
        "iss": _ISSUER,
        "exp": 9999999999,
    }
    if app_roles is not None:
        payload["app_roles"] = app_roles
    return payload


@pytest.mark.asyncio
async def test_extracts_proxy_admin_role_from_verified_jwt(monkeypatch, entra_env):
    """A tenant-JWKS-verified id_token's app_roles are extracted."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, private_key.public_key(), "key-1")
    token = _mint(private_key, _payload(["proxy_admin"]), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)
    assert roles == ["proxy_admin"]


@pytest.mark.asyncio
async def test_maps_internal_user_role(monkeypatch, entra_env):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, private_key.public_key(), "key-1")
    token = _mint(private_key, _payload(["internal_user"]), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    chosen = None
    for r in roles:
        mapped = get_litellm_user_role(r)
        if mapped is not None:
            chosen = mapped
            break

    assert chosen == LitellmUserRoles.INTERNAL_USER


@pytest.mark.asyncio
async def test_maps_proxy_admin_viewer_role(monkeypatch, entra_env):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, private_key.public_key(), "key-1")
    token = _mint(private_key, _payload(["proxy_admin_viewer"]), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)

    chosen = None
    for r in roles:
        mapped = get_litellm_user_role(r)
        if mapped is not None:
            chosen = mapped
            break

    assert chosen == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY


@pytest.mark.asyncio
async def test_no_role_in_verified_token(monkeypatch, entra_env):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, private_key.public_key(), "key-1")
    token = _mint(private_key, _payload(), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)
    assert roles == []

    default_role = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    assert default_role.value == "internal_user_viewer"


@pytest.mark.asyncio
async def test_forged_signature_yields_no_roles(monkeypatch, entra_env):
    """A token signed by a different key than the JWKS serves fails closed: no roles."""
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    real_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, real_key.public_key(), "key-1")
    token = _mint(attacker_key, _payload(["proxy_admin"]), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)
    assert roles == []


@pytest.mark.asyncio
async def test_missing_tenant_fails_closed(monkeypatch):
    """Without MICROSOFT_TENANT there is no issuer to pin; no roles are extracted."""
    monkeypatch.delenv("MICROSOFT_TENANT", raising=False)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _mint(private_key, _payload(["proxy_admin"]), "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)
    assert roles == []


@pytest.mark.asyncio
async def test_wrong_audience_yields_no_roles(monkeypatch, entra_env):
    """A token minted for a different app (aud) must not grant roles here."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _patch_jwks(monkeypatch, private_key.public_key(), "key-1")
    payload = _payload(["proxy_admin"])
    payload["aud"] = "some-other-app"
    token = _mint(private_key, payload, "key-1")

    roles = await MicrosoftSSOHandler.get_app_roles_from_id_token(token)
    assert roles == []
