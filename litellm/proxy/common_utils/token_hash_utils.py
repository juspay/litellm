"""Token digests for the LiteLLM_VerificationTokenTable and spend logs.

Historically these were bare SHA-256 hex digests. A bare digest is a
deterministic global identifier: identical keys hash identically in every
environment, and a database leak lets anyone confirm a guessed key offline.
When ``LITELLM_SALT_KEY`` is set, digests are HMAC-SHA256 keyed with it (a
server-side pepper), so a DB dump alone reveals nothing about the tokens and
cross-environment digests do not collide.

Once the pepper is configured, lookups fail closed: only the peppered digest
is accepted. Rows written before the pepper was configured still authenticate
only while the operator explicitly keeps ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS``
set (a migration window), after which it must be removed so a leaked bare
SHA-256 digest can no longer authenticate its key.
"""

import hashlib
import hmac


def _pepper() -> str | None:
    import os

    return os.getenv("LITELLM_SALT_KEY")


def _legacy_digests_allowed() -> bool:
    import os

    return os.getenv("LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS", "").lower() in ("1", "true")


def legacy_hash_token(token: str) -> str:
    """The pre-pepper digest (bare SHA-256). Kept for read-path fallback."""
    return hashlib.sha256(token.encode()).hexdigest()


def hash_token(token: str) -> str:
    """Digest a bearer token for storage/lookup.

    HMAC-SHA256 under ``LITELLM_SALT_KEY`` when the pepper is configured,
    otherwise the legacy bare SHA-256 (deployments without a salt key are
    byte-for-byte unchanged).
    """
    pepper = _pepper()
    if pepper is None:
        return legacy_hash_token(token)
    return hmac.new(pepper.encode(), token.encode(), hashlib.sha256).hexdigest()


def hash_token_candidates(token: str) -> tuple[str, ...]:
    """Every digest a stored row for ``token`` may carry, best guess first.

    On unpeppered deployments this is the single legacy digest. On peppered
    deployments it is the peppered digest alone, unless the operator has set
    ``LITELLM_ALLOW_LEGACY_TOKEN_DIGESTS`` to also accept rows written before
    the pepper was configured.
    """
    if _pepper() is None:
        return (legacy_hash_token(token),)
    if _legacy_digests_allowed():
        return (hash_token(token), legacy_hash_token(token))
    return (hash_token(token),)
