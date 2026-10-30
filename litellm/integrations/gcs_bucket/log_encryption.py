"""Client-side hybrid encryption for GCS log payloads.

A random AES-256-GCM data key encrypts each payload; the data key is wrapped
with an RSA-OAEP (SHA-256) public key exported from Cloud KMS. Only the
holder of the KMS private key can unwrap via asymmetricDecrypt, so the proxy
host can write logs it can never read back. Envelope format matches the
GCP KMS hybrid scheme documented for the log-decryption tooling:

{
  "kms":              {"project", "location", "key_ring", "key", "key_version"},
  "original_filename": "<gcs object name>",
  "metadata":          {<non-sensitive lookup fields, e.g. log_type, correlation_id>},
  "encryption":       {"algorithm": "AES-256-GCM", "nonce": b64},
  "wrapped_key":      {"algorithm": "RSA-OAEP-3072-SHA256", "ciphertext": b64},
  "data":             {"encoding": "base64", "ciphertext": b64}
}

Encryption is opt-in: it is enabled when ``GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE``
names the KMS key version
(projects/.../locations/.../keyRings/.../cryptoKeys/.../cryptoKeyVersions/N).
The public key is fetched from the Cloud KMS API at startup via application
default credentials (workload identity); no key material is deployed to the
host. ``GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH`` optionally overrides this with a
local PEM file for offline development and tests.
"""

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH = "GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH"
GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE = "GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE"

_AES_KEY_BYTES = 32
_GCM_NONCE_BYTES = 12

_KMS_RESOURCE_PATTERN = re.compile(
    r"^projects/(?P<project>[^/]+)"
    r"/locations/(?P<location>[^/]+)"
    r"/keyRings/(?P<key_ring>[^/]+)"
    r"/cryptoKeys/(?P<key>[^/]+)"
    r"/cryptoKeyVersions/(?P<key_version>[^/]+)$"
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@dataclass(frozen=True, slots=True)
class KMSKeyReference:
    project: str
    location: str
    key_ring: str
    key: str
    key_version: str

    @classmethod
    def from_resource_name(cls, resource_name: str) -> "KMSKeyReference":
        match = _KMS_RESOURCE_PATTERN.match(resource_name.strip())
        if match is None:
            raise ValueError(
                f"{GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE} must be a full KMS key version resource "
                f"(projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>/cryptoKeyVersions/<v>), "
                f"got: {resource_name!r}"
            )
        return cls(**match.groupdict())


_KMS_PUBLIC_KEY_URL = "https://cloudkms.googleapis.com/v1/{}/publicKey"


def _fetch_kms_public_key(resource_name: str) -> bytes:
    import google.auth
    import google.auth.transport.requests
    from google.auth.exceptions import GoogleAuthError
    from requests.exceptions import RequestException

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        session = google.auth.transport.requests.AuthorizedSession(credentials)
        response = session.get(_KMS_PUBLIC_KEY_URL.format(resource_name), timeout=10)
    except (GoogleAuthError, RequestException) as e:
        raise ValueError(f"could not fetch Cloud KMS public key for {resource_name!r}: {e}") from e
    if response.status_code != 200:
        raise ValueError(
            f"Cloud KMS getPublicKey failed for {resource_name!r}: HTTP {response.status_code} {response.text}"
        )
    pem_text = response.json().get("pem")
    if not pem_text:
        raise ValueError(f"Cloud KMS getPublicKey response for {resource_name!r} had no pem field")
    return pem_text.encode("utf-8")


@dataclass(frozen=True, slots=True)
class LogEncryptionConfig:
    public_key: rsa.RSAPublicKey
    kms_key: KMSKeyReference

    @classmethod
    def from_env(cls) -> "LogEncryptionConfig | None":
        public_key_path = os.getenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH)
        kms_resource = os.getenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE)
        if not public_key_path and not kms_resource:
            return None
        if not kms_resource:
            raise ValueError(
                f"{GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE} must name the KMS key version "
                "(projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>/cryptoKeyVersions/<v>)"
            )
        if public_key_path:
            pem_bytes = Path(public_key_path).read_bytes()
        else:
            pem_bytes = _fetch_kms_public_key(kms_resource)
        public_key = serialization.load_pem_public_key(pem_bytes)
        if not isinstance(public_key, rsa.RSAPublicKey):
            source = public_key_path or kms_resource
            raise ValueError(f"{source} must contain an RSA public key")
        return cls(public_key=public_key, kms_key=KMSKeyReference.from_resource_name(kms_resource))


def try_load_encryption_config_from_env() -> "tuple[LogEncryptionConfig | None, str | None]":
    try:
        return LogEncryptionConfig.from_env(), None
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm) as e:
        return None, str(e)


def encrypt_log_payload(
    payload: str,
    config: LogEncryptionConfig,
    object_name: str,
    metadata: "dict[str, str] | None" = None,
) -> str:
    aes_key = os.urandom(_AES_KEY_BYTES)
    nonce = os.urandom(_GCM_NONCE_BYTES)
    ciphertext = AESGCM(aes_key).encrypt(nonce, payload.encode("utf-8"), None)
    wrapped_key = config.public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    envelope = {
        "kms": {
            "project": config.kms_key.project,
            "location": config.kms_key.location,
            "key_ring": config.kms_key.key_ring,
            "key": config.kms_key.key,
            "key_version": config.kms_key.key_version,
        },
        "original_filename": object_name,
        "metadata": metadata or {},
        "encryption": {"algorithm": "AES-256-GCM", "nonce": _b64(nonce)},
        "wrapped_key": {
            "algorithm": f"RSA-OAEP-{config.public_key.key_size}-SHA256",
            "ciphertext": _b64(wrapped_key),
        },
        "data": {"encoding": "base64", "ciphertext": _b64(ciphertext)},
    }
    return json.dumps(envelope)


def encrypted_object_name(object_name: str) -> str:
    if object_name.endswith(".json"):
        return object_name[: -len(".json")] + ".enc.json"
    return object_name + ".enc.json"
