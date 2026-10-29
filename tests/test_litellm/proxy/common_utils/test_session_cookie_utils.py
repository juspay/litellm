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
