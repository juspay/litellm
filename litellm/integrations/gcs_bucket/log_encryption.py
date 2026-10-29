"""Client-side hybrid encryption for GCS log payloads.

A random AES-256-GCM data key encrypts each payload; the data key is wrapped
with an RSA-OAEP (SHA-256) public key exported from Cloud KMS. Only the
holder of the KMS private key can unwrap via asymmetricDecrypt, so the proxy
host can write logs it can never read back. Envelope format matches the
GCP KMS hybrid scheme documented for the log-decryption tooling:

{
  "kms":              {"project", "location", "key_ring", "key", "key_version"},
  "original_filename": "<gcs object name>",
  "encryption":       {"algorithm": "AES-256-GCM", "nonce": b64},
  "wrapped_key":      {"algorithm": "RSA-OAEP-3072-SHA256", "ciphertext": b64},
  "data":             {"encoding": "base64", "ciphertext": b64}
}

Encryption is opt-in: it is enabled only when
``GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH`` is set, in which case
``GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE`` must also name the KMS key version
(projects/.../locations/.../keyRings/.../cryptoKeys/.../cryptoKeyVersions/N).
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


@dataclass(frozen=True, slots=True)
class LogEncryptionConfig:
    public_key: rsa.RSAPublicKey
    kms_key: KMSKeyReference

    @classmethod
    def from_env(cls) -> "LogEncryptionConfig | None":
        public_key_path = os.getenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH)
        if not public_key_path:
            return None
        kms_resource = os.getenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE)
        if not kms_resource:
            raise ValueError(
                f"{GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH} is set, so "
                f"{GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE} must also be set"
            )
        public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError(f"{GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH} must contain an RSA public key")
        return cls(public_key=public_key, kms_key=KMSKeyReference.from_resource_name(kms_resource))


def try_load_encryption_config_from_env() -> "tuple[LogEncryptionConfig | None, str | None]":
    try:
        return LogEncryptionConfig.from_env(), None
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm) as e:
        return None, str(e)


def encrypt_log_payload(payload: str, config: LogEncryptionConfig, object_name: str) -> str:
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
