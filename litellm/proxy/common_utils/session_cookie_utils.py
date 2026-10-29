"""Hardened Set-Cookie handling for the admin UI session JWT.

The cookie value is an HS256 JWT signed with the proxy master key that carries
the user's virtual key and user_role — an admin credential. Every login path
(/login, /v2/login, /v3/login/exchange, and the SSO callback) sets it through
:func:`set_ui_session_cookie` so the attributes can never drift apart the way
they did when each site called ``set_cookie(key="token", ...)`` bare.
"""

from litellm.constants import LITELLM_UI_SESSION_DURATION


def set_ui_session_cookie(response, token: str, request: object | None = None) -> None:
    """Set the UI session JWT cookie with hardened attributes on any response type.

    The cookie gets the same protections the OAuth-state cookie already has:
    Secure (unless the request provably arrived over plain HTTP, e.g. local
    dev), SameSite=Lax, and a Max-Age matching LITELLM_UI_SESSION_DURATION.

    HttpOnly is opt-in via LITELLM_UI_SESSION_COOKIE_HTTPONLY=true. It stays
    off by default because the dashboard's JS must read the token from
    document.cookie when a reverse proxy (e.g. nginx-ingress) rewrites
    server-set cookies — the JS fallback path in
    ui/litellm-dashboard/src/utils/cookieUtils.ts. Deployments where JS
    never needs the cookie should enable it.
    """
    from starlette.datastructures import URL

    from litellm.litellm_core_utils.duration_parser import duration_in_seconds
    from litellm.secret_managers.main import get_secret_bool

    secure = True
    request_url = getattr(request, "url", None)
    if isinstance(request_url, URL):
        secure = request_url.scheme == "https"
    response.set_cookie(
        key="token",
        value=token,
        max_age=duration_in_seconds(str(LITELLM_UI_SESSION_DURATION)),
        secure=secure,
        samesite="lax",
        httponly=get_secret_bool("LITELLM_UI_SESSION_COOKIE_HTTPONLY"),
    )
