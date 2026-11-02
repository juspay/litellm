"""Hardened Set-Cookie handling for the admin UI session JWT.

The cookie value is an HS256 JWT signed with a key derived (via HKDF, with a
UI-session-specific info string) from the proxy master key that carries the
user's virtual key and user_role — an admin credential. Every login path
(/login, /v2/login, /v3/login/exchange, and the SSO callback) sets it through
:func:`set_ui_session_cookie` so the attributes can never drift apart the way
they did when each site called ``set_cookie(key="token", ...)`` bare.
"""

from litellm.constants import LITELLM_UI_SESSION_DURATION

# HKDF info string binding the derived key to UI-session signing, so the same
# master key can never produce this HMAC key for another purpose (and vice
# versa: an at-rest or byok key can't verify UI sessions).
_UI_SESSION_HKDF_INFO = b"litellm-ui-session-jwt-v1"
_BYOK_SESSION_HKDF_INFO = b"litellm-byok-session-jwt-v1"


def _derive_session_hkdf_key(master_key: str, info: bytes) -> bytes:
    """Derive a purpose-bound HMAC key from the master key via HKDF-SHA256.

    The master key is a bearer credential operators paste into terminals, CI,
    and helm values; using it directly as an HS256 secret means any place the
    JWT-signing material leaks also compromises every session, and any
    master-key disclosure yields forgeable admin session tokens. HKDF with a
    per-purpose info string gives key separation with a deterministic (rotation
    stable) derivation, so no new secret needs to be distributed and a key
    derived for one purpose can never verify a token for another.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(master_key.encode())


def _ui_session_signing_key(master_key: str) -> bytes:
    return _derive_session_hkdf_key(master_key, _UI_SESSION_HKDF_INFO)


def encode_byok_session_jwt(payload: dict, master_key: str) -> str:
    """Sign a BYOK MCP session JWT with a key distinct from UI sessions."""
    import jwt as pyjwt

    return pyjwt.encode(payload, _derive_session_hkdf_key(master_key, _BYOK_SESSION_HKDF_INFO), algorithm="HS256")


def decode_byok_session_jwt(token: str, master_key: str) -> dict:
    """Verify a BYOK MCP session JWT signed by :func:`encode_byok_session_jwt`."""
    import jwt as pyjwt

    return pyjwt.decode(
        token,
        _derive_session_hkdf_key(master_key, _BYOK_SESSION_HKDF_INFO),
        algorithms=["HS256"],
    )


def encode_ui_session_jwt(payload: dict, master_key: str) -> str:
    """Sign a UI-session JWT with the HKDF-derived session key (HS256)."""
    import jwt as pyjwt

    return pyjwt.encode(payload, _ui_session_signing_key(master_key), algorithm="HS256")


def decode_ui_session_jwt(token: str, master_key: str, verify_exp: bool = True, require_exp: bool = False) -> dict:
    """Verify a UI-session JWT signed by :func:`encode_ui_session_jwt`.

    Raises the PyJWT exceptions on any verification failure; callers decide
    how to map them to HTTP responses. ``require_exp`` additionally rejects
    tokens that omit the expiry claim entirely.
    """
    import jwt as pyjwt

    options: dict = {"verify_aud": False, "require": ["exp"] if require_exp else []}
    if not verify_exp:
        options["verify_exp"] = False
    return pyjwt.decode(
        token,
        _ui_session_signing_key(master_key),
        algorithms=["HS256"],
        options=options,
    )


def set_ui_session_cookie(response, token: str, request: object | None = None) -> None:
    """Set the UI session JWT cookie with hardened attributes on any response type.

    The cookie gets the same protections the OAuth-state cookie already has:
    Secure (unless the request provably arrived over plain HTTP, e.g. local
    dev), SameSite=Lax, and a Max-Age matching LITELLM_UI_SESSION_DURATION.

    HttpOnly is decided by how the response delivers the token:

    - JSON responses carry the token in the body, so the dashboard's JS
      already has it (and mirrors it into its own non-HttpOnly cookie and
      sessionStorage). The server-set cookie is a redundant copy that XSS
      must not be able to steal, so it defaults to HttpOnly.
    - Redirect responses (SSO callback) deliver the token ONLY through this
      cookie — the dashboard reads it from document.cookie on landing — so
      HttpOnly would break SSO login. Those default to JS-readable unless
      LITELLM_UI_SESSION_COOKIE_HTTPONLY=false is overridden the other way.

    LITELLM_UI_SESSION_COOKIE_HTTPONLY=true forces HttpOnly on both shapes
    (for deployments whose reverse proxy rewrites cookies and whose dashboard
    never needs the server-set cookie); "false" forces JS-readable on both.
    """
    from fastapi.responses import RedirectResponse
    from starlette.datastructures import URL

    from litellm.litellm_core_utils.duration_parser import duration_in_seconds
    from litellm.secret_managers.main import get_secret_bool

    secure = True
    request_url = getattr(request, "url", None)
    if isinstance(request_url, URL):
        secure = request_url.scheme == "https"
    is_redirect = isinstance(response, RedirectResponse)
    htt_only = get_secret_bool("LITELLM_UI_SESSION_COOKIE_HTTPONLY")
    if htt_only is None:
        htt_only = not is_redirect
    response.set_cookie(
        key="token",
        value=token,
        max_age=duration_in_seconds(str(LITELLM_UI_SESSION_DURATION)),
        secure=secure,
        samesite="lax",
        httponly=htt_only,
    )
