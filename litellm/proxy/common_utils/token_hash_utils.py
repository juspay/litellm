"""Token digests for the LiteLLM_VerificationTokenTable and spend logs.

Historically these were bare SHA-256 hex digests. A bare digest is a
deterministic global identifier: identical keys hash identically in every
environment, and a database leak lets anyone confirm a guessed key offline.
When ``LITELLM_SALT_KEY`` is set, digests are HMAC-SHA256 keyed with it (a
server-side pepper), so a DB dump alone reveals nothing about the tokens and
cross-environment digests do not collide.

Legacy SHA-256 rows still authenticate: read paths try the peppered digest
first and fall back to :func:`legacy_hash_token` via
:func:`hash_token_candidates`.
"""

import hashlib
import hmac


def _pepper() -> str | None:
    import os

    return os.getenv("LITELLM_SALT_KEY")


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
    deployments it is the peppered digest plus the legacy digest so lookups
    still find rows written before the pepper was configured.
    """
    if _pepper() is None:
        return (legacy_hash_token(token),)
    return (hash_token(token), legacy_hash_token(token))
