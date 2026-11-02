"""Token digests for the LiteLLM_VerificationTokenTable and spend logs.

Historically these were bare SHA-256 hex digests. A bare digest is a
deterministic global identifier: identical keys hash identically in every
environment, and a database leak lets anyone confirm a guessed key offline.
When ``LITELLM_SALT_KEY`` is set, digests are HMAC-SHA256 keyed with it (a
server-side pepper), so a DB dump alone reveals nothing about the tokens and
cross-environment digests do not collide.

Peppered deployments fail closed on both sides: only the peppered digest is
written, and only the peppered digest is accepted on lookup. Rows written
before the pepper was configured still authenticate only while the operator
explicitly keeps ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`` set (a migration
window), after which it must be removed so a leaked bare SHA-256 digest can
no longer authenticate its key.

Unpeppered deployments run in a warned, explicitly-opted-in legacy mode: the
bare digest is only written when ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`` is
set, and a startup warning fires when it is not (the deployment must set
``LITELLM_SALT_KEY`` to mint or authenticate virtual keys under a pepper).
"""

import functools
import hashlib
import hmac

from litellm._logging import verbose_proxy_logger


def _pepper() -> str | None:
    import os

    return os.getenv("LITELLM_SALT_KEY")


def _legacy_digests_allowed() -> bool:
    import os

    return os.getenv("LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS", "").lower() in ("1", "true")


def legacy_hash_token(token: str) -> str:
    """The pre-pepper digest (bare SHA-256). Kept for read-path fallback."""
    return hashlib.sha256(token.encode()).hexdigest()


@functools.cache
def _warn_unpeppered_legacy_mode_once() -> None:
    verbose_proxy_logger.warning(
        "LITELLM_SALT_KEY is not set; token digests are bare SHA-256 and a database "
        "leak would allow offline key confirmation. Set LITELLM_SALT_KEY (see "
        "https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key) or set "
        "LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1 to explicitly accept this legacy mode."
    )


def hash_token(token: str) -> str:
    """Digest a bearer token for storage/lookup.

    HMAC-SHA256 under ``LITELLM_SALT_KEY`` when the pepper is configured. On
    unpeppered deployments the legacy bare SHA-256 digest is returned only in
    the explicit ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`` migration mode; a
    fresh unpeppered deployment fails closed (raises) rather than silently
    minting unsalted digests.
    """
    pepper = _pepper()
    if pepper is not None:
        return hmac.new(pepper.encode(), token.encode(), hashlib.sha256).hexdigest()
    if _legacy_digests_allowed():
        _warn_unpeppered_legacy_mode_once()
        return legacy_hash_token(token)
    raise ValueError(
        "LITELLM_SALT_KEY is not set; refusing to mint an unpeppered token digest. "
        "Set LITELLM_SALT_KEY or set LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1 to "
        "explicitly accept the legacy bare-SHA-256 mode."
    )


def hash_token_candidates(token: str) -> tuple[str, ...]:
    """Every digest a stored row for ``token`` may carry, best guess first.

    On unpeppered legacy-mode deployments this is the single legacy digest. On
    peppered deployments it is the peppered digest alone, unless the operator
    has set ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`` to also accept rows written
    before the pepper was configured.
    """
    if _pepper() is None:
        if _legacy_digests_allowed():
            _warn_unpeppered_legacy_mode_once()
            return (legacy_hash_token(token),)
        raise ValueError(
            "LITELLM_SALT_KEY is not set; refusing to look up token digests. "
            "Set LITELLM_SALT_KEY or set LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS=1 to "
            "explicitly accept the legacy bare-SHA-256 mode."
        )
    if _legacy_digests_allowed():
        return (hash_token(token), legacy_hash_token(token))
    return (hash_token(token),)
