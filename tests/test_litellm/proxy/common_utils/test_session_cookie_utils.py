import time

import jwt as pyjwt
import pytest
from starlette.responses import Response


def _set_cookie_header(response: Response) -> str:
    headers = response.headers.getlist("set-cookie")
    assert len(headers) == 1
    return headers[0]


def test_session_cookie_has_hardened_attributes() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import set_ui_session_cookie

    response = Response()
    set_ui_session_cookie(response, "jwt-value")
    header = _set_cookie_header(response)
    assert "token=jwt-value" in header
    assert "Secure" in header
    assert "SameSite=lax" in header
    assert "HttpOnly" not in header
    assert f"Max-Age={24 * 60 * 60}" in header


def test_session_cookie_not_secure_over_plain_http_request() -> None:
    from starlette.requests import Request

    from litellm.proxy.common_utils.session_cookie_utils import set_ui_session_cookie

    request = Request(scope={"type": "http", "scheme": "http", "method": "GET", "path": "/login", "headers": [], "query_string": b""})
    response = Response()
    set_ui_session_cookie(response, "jwt-value", request=request)
    header = _set_cookie_header(response)
    assert "Secure" not in header


def test_session_cookie_httponly_opt_in(monkeypatch) -> None:
    from litellm.proxy.common_utils.session_cookie_utils import set_ui_session_cookie

    monkeypatch.setenv("LITELLM_UI_SESSION_COOKIE_HTTPONLY", "true")
    response = Response()
    set_ui_session_cookie(response, "jwt-value")
    assert "HttpOnly" in _set_cookie_header(response)


def test_session_cookie_custom_duration(monkeypatch) -> None:
    from litellm.proxy.common_utils.session_cookie_utils import set_ui_session_cookie

    monkeypatch.setattr(
        "litellm.proxy.common_utils.session_cookie_utils.LITELLM_UI_SESSION_DURATION", "30m"
    )
    response = Response()
    set_ui_session_cookie(response, "jwt-value")
    assert "Max-Age=1800" in _set_cookie_header(response)


MASTER_KEY = "sk-master-1234"
UI_PAYLOAD = {"user_id": "u1", "user_role": "proxy_admin", "login_method": "sso"}


def _encode_ui_session(payload: dict) -> str:
    from litellm.proxy.common_utils.session_cookie_utils import encode_ui_session_jwt

    return encode_ui_session_jwt(payload, MASTER_KEY)


def test_ui_session_jwt_round_trip() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import decode_ui_session_jwt

    token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) + 60})
    claims = decode_ui_session_jwt(token, MASTER_KEY)
    assert claims["user_id"] == "u1"
    assert claims["user_role"] == "proxy_admin"
    assert claims["exp"] > int(time.time())


def test_ui_session_jwt_not_verifiable_with_raw_master_key() -> None:
    token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) + 60})
    with pytest.raises(pyjwt.InvalidSignatureError):
        pyjwt.decode(token, MASTER_KEY, algorithms=["HS256"], options={"verify_aud": False})


def test_ui_session_jwt_rejects_wrong_master_key() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import decode_ui_session_jwt

    token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) + 60})
    with pytest.raises(pyjwt.InvalidSignatureError):
        decode_ui_session_jwt(token, "sk-other-key")


def test_ui_session_jwt_expired_token_rejected() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import decode_ui_session_jwt

    token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) - 60})
    with pytest.raises(pyjwt.ExpiredSignatureError):
        decode_ui_session_jwt(token, MASTER_KEY)


def test_ui_session_jwt_verify_exp_false_accepts_expired() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import decode_ui_session_jwt

    token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) - 60})
    claims = decode_ui_session_jwt(token, MASTER_KEY, verify_exp=False)
    assert claims["user_id"] == "u1"


def test_ui_session_jwt_require_exp_rejects_missing_expiry() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import decode_ui_session_jwt

    token = _encode_ui_session(dict(UI_PAYLOAD))
    with pytest.raises(pyjwt.MissingRequiredClaimError):
        decode_ui_session_jwt(token, MASTER_KEY, require_exp=True)


def test_byok_session_jwt_cannot_verify_ui_session_jwt() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import (
        decode_byok_session_jwt,
        decode_ui_session_jwt,
        encode_byok_session_jwt,
    )

    ui_token = _encode_ui_session({**UI_PAYLOAD, "exp": int(time.time()) + 60})
    with pytest.raises(pyjwt.InvalidSignatureError):
        decode_byok_session_jwt(ui_token, MASTER_KEY)

    byok_token = encode_byok_session_jwt({"user_id": "u1"}, MASTER_KEY)
    with pytest.raises(pyjwt.InvalidSignatureError):
        decode_ui_session_jwt(byok_token, MASTER_KEY, verify_exp=False)


def test_hkdf_keys_differ_per_purpose_and_per_master_key() -> None:
    from litellm.proxy.common_utils.session_cookie_utils import (
        _BYOK_SESSION_HKDF_INFO,
        _UI_SESSION_HKDF_INFO,
        _derive_session_hkdf_key,
    )

    ui_key = _derive_session_hkdf_key(MASTER_KEY, _UI_SESSION_HKDF_INFO)
    byok_key = _derive_session_hkdf_key(MASTER_KEY, _BYOK_SESSION_HKDF_INFO)
    assert ui_key != byok_key
    assert len(ui_key) == 32
    assert _derive_session_hkdf_key("sk-other-key", _UI_SESSION_HKDF_INFO) != ui_key
