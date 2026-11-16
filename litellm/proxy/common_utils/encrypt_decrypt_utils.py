import base64
import os
import re
from typing import Final, Literal

from litellm._logging import verbose_proxy_logger

# Versioned ciphertext markers:
#   v2:gcm:  + base64url(nonce(12) || ciphertext || tag(16)), key = SHA-256(salt)
#   v3:hkdf: + base64url(nonce(12) || ciphertext || tag(16)), key = HKDF-SHA256(salt)
#   v4:ctx:  + <context length>:<context> + base64url(nonce || ciphertext || tag),
#             key = HKDF-SHA256(salt), AESGCM AAD = "litellm-at-rest:" + context
# Legacy XSalsa20-Poly1305 (nacl) values carry no marker; the colon in the
# prefixes can never appear in base64url(nacl output), so the prefix check is an
# unambiguous discriminator between the formats on read.
_V2_GCM_PREFIX = "v2:gcm:"
_V3_HKDF_PREFIX = "v3:hkdf:"
_V4_CTX_PREFIX = "v4:ctx:"
_VERSIONED_CIPHERTEXT_PREFIXES = (_V2_GCM_PREFIX, _V3_HKDF_PREFIX, _V4_CTX_PREFIX)


def is_versioned_ciphertext(value: object) -> bool:
    """True if ``value`` is an AES-256-GCM ciphertext (any v2/v3/v4 marker)."""
    return isinstance(value, str) and value.startswith(_VERSIONED_CIPHERTEXT_PREFIXES)


# HKDF info string binds the derived key to this exact use ("litellm at-rest
# encryption"), so the same salt can never produce this key for another purpose.
_HKDF_INFO = b"litellm-at-rest-encryption-v1"
# AAD prefix for context-bound values; the separator cannot appear inside a
# context because contexts are validated to [A-Za-z0-9_.:-]+ on both sides.
_CTX_AAD_PREFIX = b"litellm-at-rest:"
_CTX_AAD_SEPARATOR = ":"
_CTX_VALID_RE = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")

# general_settings key selecting the at-rest encryption algorithm for new writes.
# Default preserves the legacy algorithm so existing deployments are byte-for-byte
# unchanged until they explicitly opt in. Decrypt is always format-detecting, so
# flipping this flag forward (or back) never strands previously-written data.
_ENCRYPTION_ALGORITHM_SETTING = "encryption_algorithm"
_ALGO_AES_GCM = "aes-256-gcm"
_ALGO_XSALSA20 = "xsalsa20-poly1305"
# Set to bypass the production guard on the salt key (explicitly opts a
# DB-connected deployment into the master-key fallback).
_SALT_KEY_FALLBACK_ALLOWED_ENV = "LITELLM_ALLOW_SALT_KEY_FALLBACK"


def _salt_key_fallback_is_forbidden() -> bool:
    """The master-key fallback is forbidden unless explicitly opted into.

    Deriving the at-rest key from the same ``LITELLM_MASTER_KEY`` that is also
    a bearer credential means a single leaked master key decrypts every stored
    provider key / OAuth token, and rotating it silently strands every stored
    value. ``LITELLM_SALT_KEY`` is required for any at-rest encryption
    regardless of DB state; ``LITELLM_ALLOW_SALT_KEY_FALLBACK=1`` is the
    explicitly logged migration window where an operator accepts the fallback
    while the salt key has not been deployed yet.
    """
    return os.getenv(_SALT_KEY_FALLBACK_ALLOWED_ENV, "") not in ("1", "true", "True")


def _get_salt_key():
    from litellm.proxy.proxy_server import master_key

    salt_key = os.getenv("LITELLM_SALT_KEY", None)

    if salt_key is None:
        if _salt_key_fallback_is_forbidden():
            raise ValueError(
                "LITELLM_SALT_KEY is not set; refusing to derive the at-rest "
                "encryption key from LITELLM_MASTER_KEY. Set LITELLM_SALT_KEY "
                "(see https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key) "
                f"or set {_SALT_KEY_FALLBACK_ALLOWED_ENV}=1 to explicitly accept the "
                "master-key fallback during a migration window."
            )
        verbose_proxy_logger.warning(
            f"LITELLM_SALT_KEY is not set; using LITELLM_MASTER_KEY as the at-rest "
            f"encryption key because {_SALT_KEY_FALLBACK_ALLOWED_ENV} is set. A leaked "
            "master key decrypts every stored credential; deploy LITELLM_SALT_KEY "
            "and remove the flag."
        )
        salt_key = master_key

    return salt_key


def _get_encryption_algorithm() -> str:
    """
    Resolve the configured at-rest encryption algorithm for *new writes*.

    Read from ``general_settings.encryption_algorithm`` at write time. Defaults
    to AES-256-GCM + HKDF (the audited KDF) for new deployments; deployments
    that predate the switch keep writing legacy XSalsa20-Poly1305 until they
    explicitly opt in. Decrypt is always format-detecting, so either default
    reads every historical value.
    """
    try:
        from litellm.proxy.proxy_server import general_settings

        algo = general_settings.get(_ENCRYPTION_ALGORITHM_SETTING, _ALGO_AES_GCM)
    except Exception:
        # general_settings may not be importable in some contexts (e.g. SDK-only
        # use of these helpers). Use the audited default.
        return _ALGO_AES_GCM

    if isinstance(algo, str) and algo.lower() == _ALGO_XSALSA20:
        return _ALGO_XSALSA20
    return _ALGO_AES_GCM


def _derive_key(signing_key: str) -> bytes:
    """Derive a 32-byte key from the salt/master key (shared by both algorithms).

    Known limitation: this is a single-pass, unsalted ``SHA-256`` of the key, not
    a dedicated KDF (HKDF/PBKDF2). It is the *same* derivation the legacy nacl
    path already uses, so the AES path introduces no new weakness and stays
    interoperable with existing key sourcing; AES-256-GCM's per-value 12-byte
    random nonce gives the unique (key, nonce) pairs GCM requires. This
    derivation only handles ciphertext written under the ``v2:gcm:`` marker (and
    the legacy nacl format, which shares it); new AES writes use
    :func:`_derive_key_hkdf` behind the ``v3:hkdf:`` marker. Read-path compat
    for this KDF is bounded by the migration tooling: the
    ``credential_migration`` module re-encrypts ``v2:gcm:`` rows to HKDF and the
    ``check_encryption`` attestation counts them as unmigrated until they are.
    """
    import hashlib

    return hashlib.sha256(signing_key.encode()).digest()


def _derive_key_hkdf(signing_key: str) -> bytes:
    """Derive a 32-byte key with HKDF-SHA256, the audited KDF for new writes.

    HKDF's extract step concentrates the entropy of the operator's key material
    into a pseudorandom key before expansion, so the derived AES key is not a
    direct function of the master key bytes the way the single unsalted SHA-256
    pass in :func:`_derive_key` is. Values sealed under this derivation carry
    the ``v3:hkdf:`` marker so reads pick the matching KDF.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_HKDF_INFO,
    ).derive(signing_key.encode())


def _seal_aes_gcm(value: str, key: bytes, aad: bytes | None) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    # AESGCM.encrypt returns ciphertext || tag(16); wire format is nonce || that.
    return nonce + AESGCM(key).encrypt(nonce, value.encode("utf-8"), aad)


def _open_aes_gcm(sealed: bytes, key: bytes, aad: bytes | None) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    # An empty plaintext still serializes to nonce(12) || tag(16) = 28 bytes, so a
    # short/empty buffer here is a corrupt value: let AESGCM.decrypt raise and be
    # swallowed by the caller (returns None/original), same as legacy.
    return AESGCM(key).decrypt(sealed[:12], sealed[12:], aad).decode("utf-8")


def _ctx_aad(context: str) -> bytes:
    """Build the AAD for a context-bound value; raise on an invalid context.

    Contexts come from internal ``f"{table}:{field}"`` strings, never from
    request data, but they are still validated against a conservative alphabet
    so the ``:`` framing of the v4 wire format can never be confused.
    """
    if not _CTX_VALID_RE.fullmatch(context):
        raise ValueError(f"Invalid encryption context: {context!r}")
    return _CTX_AAD_PREFIX + context.encode("utf-8")


def _encrypt_aes_gcm_ctx(value: str, signing_key: str, context: str) -> str:
    """Encrypt under AES-256-GCM + HKDF, binding the ciphertext to its context.

    The context (``table:field`` the value belongs to) is both embedded in the
    v4 wire format and fed as the AES-GCM AAD, so a ciphertext lifted from one
    column cannot be replayed into another: it fails authentication there.
    """
    sealed: Final = _seal_aes_gcm(value=value, key=_derive_key_hkdf(signing_key), aad=_ctx_aad(context))
    encoded = base64.urlsafe_b64encode(sealed).decode("utf-8")
    ctx_len = len(context.encode("utf-8"))
    return f"{_V4_CTX_PREFIX}{ctx_len}{_CTX_AAD_SEPARATOR}{context}{encoded}"


def _parse_v4_value(value: str, context: str | None) -> tuple[bytes, bytes | None]:
    """Split a v4 string into (sealed, aad), validating the embedded context.

    Raises ValueError when the framing is corrupt, the embedded context does
    not match the expected one, or no expected context was supplied (fail
    closed: a context-bound value must never decrypt as context-free).
    """
    rest = value[len(_V4_CTX_PREFIX) :]
    length_str, _, tail = rest.partition(_CTX_AAD_SEPARATOR)
    if not length_str.isdigit():
        raise ValueError("Corrupt v4 ciphertext: bad context length")
    ctx_bytes = length_str and int(length_str)
    embedded = tail[:ctx_bytes].encode("utf-8")
    encoded = tail[ctx_bytes:]
    if embedded != (context or "").encode("utf-8"):
        raise ValueError(f"Ciphertext context mismatch: encrypted for {embedded!r}, read as {context!r}")
    if context is None:
        raise ValueError("Context-bound value read without a context")
    return base64.urlsafe_b64decode(encoded), _ctx_aad(context)


def _decrypt_aes_gcm(value: str, signing_key: str, context: str | None = None) -> str:
    """Decrypt a versioned ``v2:gcm:``/``v3:hkdf:``/``v4:ctx:`` string."""
    if value.startswith(_V4_CTX_PREFIX):
        sealed, aad = _parse_v4_value(value, context)
        return _open_aes_gcm(sealed=sealed, key=_derive_key_hkdf(signing_key), aad=aad)
    if value.startswith(_V3_HKDF_PREFIX):
        sealed: Final = base64.urlsafe_b64decode(value[len(_V3_HKDF_PREFIX) :])
        return _open_aes_gcm(sealed=sealed, key=_derive_key_hkdf(signing_key), aad=None)
    sealed = base64.urlsafe_b64decode(value[len(_V2_GCM_PREFIX) :])
    return _open_aes_gcm(sealed=sealed, key=_derive_key(signing_key), aad=None)


def _encrypt_aes_gcm(value: str, signing_key: str) -> str:
    """Encrypt under AES-256-GCM + HKDF and return the versioned ``v3:hkdf:`` string."""
    sealed: Final = _seal_aes_gcm(value=value, key=_derive_key_hkdf(signing_key), aad=None)
    return _V3_HKDF_PREFIX + base64.urlsafe_b64encode(sealed).decode("utf-8")


def encrypt_bearer_token(value: str, prefix: str) -> str:
    """AES-256-GCM as unpadded base64url behind ``prefix``, which is also the AAD so a token can't change kind."""
    salt_key: Final = _get_salt_key()
    if not isinstance(salt_key, str):
        raise ValueError("Set LITELLM_SALT_KEY or a master key to mint bearer tokens")
    sealed: Final = _seal_aes_gcm(value=value, key=_derive_key_hkdf(salt_key), aad=prefix.encode("utf-8"))
    return prefix + base64.urlsafe_b64encode(sealed).decode("ascii").rstrip("=")


def decrypt_bearer_token(token: str, prefix: str) -> str | None:
    """None unless ``token`` came from :func:`encrypt_bearer_token` with the same ``prefix``."""
    salt_key: Final = _get_salt_key()
    if not isinstance(salt_key, str) or not token.startswith(prefix):
        return None
    encoded: Final = token.removeprefix(prefix)
    try:
        sealed: Final = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        return _open_aes_gcm(sealed=sealed, key=_derive_key_hkdf(salt_key), aad=prefix.encode("utf-8"))
    except Exception:  # noqa: BLE001  # base64 and AES-GCM each raise their own "not a token" type
        return None


def encrypt_value_helper(
    value: str,
    new_encryption_key: str | None = None,
    context: str | None = None,
):
    """Encrypt a value for at-rest storage.

    ``context`` optionally names the ``table:field`` the value is stored under;
    when given, the ciphertext is bound to it (v4 format) so it cannot be
    transplanted into another column. Sites that cannot know their storage
    context keep the context-free v3 format.
    """
    signing_key = new_encryption_key or _get_salt_key()

    if isinstance(value, str):
        if _get_encryption_algorithm() == _ALGO_AES_GCM:
            if context is not None:
                return _encrypt_aes_gcm_ctx(value=value, signing_key=signing_key, context=context)
            return _encrypt_aes_gcm(value=value, signing_key=signing_key)

        encrypted_value = encrypt_value(value=value, signing_key=signing_key)  # type: ignore
        # Use urlsafe_b64encode for URL-safe base64 encoding (replaces + with - and / with _)
        encrypted_value = base64.urlsafe_b64encode(encrypted_value).decode("utf-8")

        return encrypted_value

    verbose_proxy_logger.debug(
        f"Invalid value type passed to encrypt_value: {type(value)} for Value: {value}\n Value must be a string"
    )
    # if it's not a string - do not encrypt it and return the value
    return value


def decrypt_value_helper(
    value: str,
    key: str,  # this is just for debug purposes, showing the k,v pair that's invalid. not a signing key.
    exception_type: Literal["debug", "error"] = "error",
    return_original_value: bool = False,
    context: str | None = None,
):
    """Decrypt an at-rest value.

    ``context`` must match the one the value was encrypted with; a context-bound
    (v4) value read with a different or missing context fails closed (returns
    None / the original, never a wrong-field plaintext).
    """
    signing_key = _get_salt_key()

    try:
        if isinstance(value, str):
            # Versioned AES-256-GCM values are detected before any base64 decode.
            # The prefix is the algorithm tag the legacy nacl format never carried.
            if is_versioned_ciphertext(value):
                return _decrypt_aes_gcm(value=value, signing_key=signing_key, context=context)

            # Try URL-safe base64 decoding first (new format)
            # Fall back to standard base64 decoding for backwards compatibility (old format)
            try:
                decoded_b64 = base64.urlsafe_b64decode(value)
            except Exception:
                # If URL-safe decoding fails, try standard base64 decoding for backwards compatibility
                decoded_b64 = base64.b64decode(value)

            value = decrypt_value(value=decoded_b64, signing_key=signing_key)  # type: ignore
            return value

        # if it's not str - do not decrypt it, return the value
        return value
    except Exception as e:
        error_message = f"Error decrypting value for key: {key}, Did your master_key/salt key change recently? \nError: {str(e)}\nSet permanent salt key - https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
        if exception_type == "debug":
            verbose_proxy_logger.debug(error_message)
            return value if return_original_value else None

        verbose_proxy_logger.debug(f"Unable to decrypt value for key: {key}, returning None")
        if return_original_value:
            return value
        else:
            verbose_proxy_logger.exception(error_message)
            # [Non-Blocking Exception. - this should not block decrypting other values]
            return None


def encrypt_value(value: str, signing_key: str):
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # encode message #
    value_bytes = value.encode("utf-8")

    encrypted = box.encrypt(value_bytes)

    return encrypted


def decrypt_value(value: bytes, signing_key: str) -> str:
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # Convert the bytes object to a string
    try:
        if len(value) == 0:
            return ""

        plaintext = box.decrypt(value)
        plaintext = plaintext.decode("utf-8")  # type: ignore
        return plaintext  # type: ignore
    except Exception as e:
        raise e
