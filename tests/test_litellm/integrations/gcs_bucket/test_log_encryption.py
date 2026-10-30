import base64
import json

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from litellm.integrations.gcs_bucket import log_encryption
from litellm.integrations.gcs_bucket.gcs_logger import (
    ProductionGCSLogger,
    _gcs_filename_user_part,
)
from litellm.integrations.gcs_bucket.log_encryption import (
    GCS_LOG_ENCRYPTION_KMS_KEY,
    GCS_LOG_ENCRYPTION_KMS_KEY_RING,
    GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE,
    GCS_LOG_ENCRYPTION_KMS_KEY_VERSION,
    GCS_LOG_ENCRYPTION_KMS_LOCATION,
    GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH,
    KMSKeyReference,
    LogEncryptionConfig,
    encrypt_log_payload,
    encrypted_object_name,
    try_load_encryption_config_from_env,
)

KMS_RESOURCE = "projects/test-project/locations/asia-south1/keyRings/test-ring/cryptoKeys/test-key/cryptoKeyVersions/1"
PLAINTEXT_MARKER = "super-secret-marker-9f27b1"


@pytest.fixture()
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def config(private_key: rsa.RSAPrivateKey) -> LogEncryptionConfig:
    return LogEncryptionConfig(
        public_key=private_key.public_key(),
        kms_key=KMSKeyReference.from_resource_name(KMS_RESOURCE),
    )


def _decrypt_envelope(envelope: dict, private_key: rsa.RSAPrivateKey) -> bytes:
    aes_key = private_key.decrypt(
        base64.b64decode(envelope["wrapped_key"]["ciphertext"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return AESGCM(aes_key).decrypt(
        base64.b64decode(envelope["encryption"]["nonce"]),
        base64.b64decode(envelope["data"]["ciphertext"]),
        None,
    )


def _set_encryption_env(monkeypatch: pytest.MonkeyPatch, tmp_path, private_key: rsa.RSAPrivateKey) -> None:
    pem_path = tmp_path / "public-key.pem"
    pem_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, str(pem_path))
    monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)


def _make_logger(monkeypatch: pytest.MonkeyPatch, tmp_path, private_key: rsa.RSAPrivateKey) -> ProductionGCSLogger:
    _set_encryption_env(monkeypatch, tmp_path, private_key)
    return ProductionGCSLogger()


class _GCSBaseDouble:
    def __init__(self) -> None:
        self.calls = []

    async def construct_request_headers(self, service_account_json, vertex_instance):
        return {}

    async def _log_json_data_on_gcs(self, headers, bucket_name, object_name, logging_payload):
        self.calls.append(
            {
                "headers": headers,
                "bucket_name": bucket_name,
                "object_name": object_name,
                "logging_payload": logging_payload,
            }
        )


def _uploaded(log_double: _GCSBaseDouble):
    assert len(log_double.calls) == 1
    call = log_double.calls[0]
    return call["object_name"], call["logging_payload"]


class TestKMSKeyReference:
    def test_parses_full_resource_name(self):
        ref = KMSKeyReference.from_resource_name(KMS_RESOURCE)
        assert ref.project == "test-project"
        assert ref.location == "asia-south1"
        assert ref.key_ring == "test-ring"
        assert ref.key == "test-key"
        assert ref.key_version == "1"

    @pytest.mark.parametrize(
        "resource",
        [
            "projects/p/locations/l/keyRings/r/cryptoKeys/k",
            "projects/p/locations/l/keyRings/r/cryptoKeys/k/cryptoKeyVersions/",
            "test-key",
            "",
        ],
    )
    def test_rejects_malformed_resource_names(self, resource: str):
        with pytest.raises(ValueError):
            KMSKeyReference.from_resource_name(resource)


class TestLoadConfigFromEnv:
    def test_disabled_when_public_key_path_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, raising=False)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, raising=False)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_KMS_KEY, raising=False)
        config, error = try_load_encryption_config_from_env()
        assert config is None
        assert error is None

    def test_error_when_kms_resource_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, config: LogEncryptionConfig
    ):
        pem_path = tmp_path / "public-key.pem"
        pem_path.write_bytes(
            config.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, str(pem_path))
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, raising=False)
        loaded, error = try_load_encryption_config_from_env()
        assert loaded is None
        assert error is not None

    def test_error_when_public_key_file_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, str(tmp_path / "nope.pem"))
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        loaded, error = try_load_encryption_config_from_env()
        assert loaded is None
        assert error is not None

    def test_error_when_key_is_not_rsa(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        ec_pem_path = tmp_path / "ec-public.pem"
        ec_pem_path.write_bytes(
            ec.generate_private_key(ec.SECP256R1())
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, str(ec_pem_path))
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        loaded, error = try_load_encryption_config_from_env()
        assert loaded is None
        assert error is not None

    def test_loads_valid_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path, private_key):
        _set_encryption_env(monkeypatch, tmp_path, private_key)
        loaded, error = try_load_encryption_config_from_env()
        assert error is None
        assert loaded is not None
        assert loaded.kms_key.key == "test-key"
        assert loaded.public_key.key_size == private_key.key_size

    def test_kms_resource_only_fetches_public_key_from_kms(self, monkeypatch: pytest.MonkeyPatch, private_key):
        fetched_with = []

        def fake_fetch(resource_name: str) -> bytes:
            fetched_with.append(resource_name)
            return private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        monkeypatch.setattr(log_encryption, "_fetch_kms_public_key", fake_fetch)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, raising=False)
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        loaded, error = try_load_encryption_config_from_env()
        assert error is None
        assert fetched_with == [KMS_RESOURCE]
        assert loaded is not None
        assert loaded.kms_key.key == "test-key"
        assert loaded.public_key.public_numbers() == private_key.public_key().public_numbers()

    def test_pem_file_wins_over_fetch_when_both_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path, private_key):
        def fail_fetch(resource_name: str) -> bytes:
            raise AssertionError("fetch must not be called when PEM file is set")

        monkeypatch.setattr(log_encryption, "_fetch_kms_public_key", fail_fetch)
        _set_encryption_env(monkeypatch, tmp_path, private_key)
        loaded, error = try_load_encryption_config_from_env()
        assert error is None
        assert loaded is not None

    def test_error_when_fetch_fails(self, monkeypatch: pytest.MonkeyPatch):
        def fail_fetch(resource_name: str) -> bytes:
            raise ValueError("no credentials")

        monkeypatch.setattr(log_encryption, "_fetch_kms_public_key", fail_fetch)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, raising=False)
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        loaded, error = try_load_encryption_config_from_env()
        assert loaded is None
        assert error is not None
        assert "no credentials" in error

    def test_error_when_fetched_key_is_not_rsa(self, monkeypatch: pytest.MonkeyPatch):
        ec_pem = (
            ec.generate_private_key(ec.SECP256R1())
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setattr(log_encryption, "_fetch_kms_public_key", lambda resource_name: ec_pem)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, raising=False)
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        loaded, error = try_load_encryption_config_from_env()
        assert loaded is None
        assert error is not None
        assert "must contain an RSA public key" in error


class TestFetchKMSPublicKey:
    def test_fetches_pem_via_authorized_session(self, monkeypatch: pytest.MonkeyPatch, private_key):
        import google.auth
        import google.auth.transport.requests as google_requests

        pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        refreshed = []
        captured = {}

        class FakeCredentials:
            def refresh(self, request):
                refreshed.append(request)

        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"pem": pem.decode("utf-8")}

        class FakeSession:
            def __init__(self, credentials):
                captured["credentials"] = credentials

            def get(self, url, timeout):
                captured["url"] = url
                captured["timeout"] = timeout
                return FakeResponse()

        monkeypatch.setattr(google.auth, "default", lambda scopes=None: (FakeCredentials(), "test-project"))
        monkeypatch.setattr(google_requests, "AuthorizedSession", FakeSession)

        result = log_encryption._fetch_kms_public_key(KMS_RESOURCE)

        assert captured["url"] == f"https://cloudkms.googleapis.com/v1/{KMS_RESOURCE}/publicKey"
        assert captured["timeout"] == 10
        assert len(refreshed) == 1
        assert result == pem
        assert isinstance(serialization.load_pem_public_key(result), rsa.RSAPublicKey)

    def test_raises_on_http_error(self, monkeypatch: pytest.MonkeyPatch):
        import google.auth
        import google.auth.transport.requests as google_requests

        class FakeCredentials:
            def refresh(self, request):
                pass

        class FakeSession:
            def __init__(self, credentials):
                pass

            def get(self, url, timeout):
                class FakeResponse:
                    status_code = 403
                    text = "Forbidden"

                return FakeResponse()

        monkeypatch.setattr(google.auth, "default", lambda scopes=None: (FakeCredentials(), "test-project"))
        monkeypatch.setattr(google_requests, "AuthorizedSession", FakeSession)

        with pytest.raises(ValueError) as excinfo:
            log_encryption._fetch_kms_public_key(KMS_RESOURCE)
        assert "403" in str(excinfo.value)
        assert KMS_RESOURCE in str(excinfo.value)

    def test_raises_when_response_has_no_pem(self, monkeypatch: pytest.MonkeyPatch):
        import google.auth
        import google.auth.transport.requests as google_requests

        class FakeCredentials:
            def refresh(self, request):
                pass

        class FakeSession:
            def __init__(self, credentials):
                pass

            def get(self, url, timeout):
                class FakeResponse:
                    status_code = 200
                    text = ""

                    def json(self):
                        return {}

                return FakeResponse()

        monkeypatch.setattr(google.auth, "default", lambda scopes=None: (FakeCredentials(), "test-project"))
        monkeypatch.setattr(google_requests, "AuthorizedSession", FakeSession)

        with pytest.raises(ValueError) as excinfo:
            log_encryption._fetch_kms_public_key(KMS_RESOURCE)
        assert "pem" in str(excinfo.value)

    def test_wraps_google_auth_errors(self, monkeypatch: pytest.MonkeyPatch):
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        def raise_default(scopes=None):
            raise DefaultCredentialsError("nope")

        monkeypatch.setattr(google.auth, "default", raise_default)

        with pytest.raises(ValueError) as excinfo:
            log_encryption._fetch_kms_public_key(KMS_RESOURCE)
        assert "nope" in str(excinfo.value)
        assert KMS_RESOURCE in str(excinfo.value)


class TestShortKeyNameResolution:
    def _fake_fetch(self, monkeypatch: pytest.MonkeyPatch, private_key):
        fetched_with = []

        def fake_fetch(resource_name: str) -> bytes:
            fetched_with.append(resource_name)
            return private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        monkeypatch.setattr(log_encryption, "_fetch_kms_public_key", fake_fetch)
        return fetched_with

    def _set_short_form_env(self, monkeypatch: pytest.MonkeyPatch, key_name: str):
        for env_var in (
            GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH,
            GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE,
            GCS_LOG_ENCRYPTION_KMS_LOCATION,
            GCS_LOG_ENCRYPTION_KMS_KEY_RING,
            GCS_LOG_ENCRYPTION_KMS_KEY_VERSION,
        ):
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY, key_name)

    def test_short_key_name_resolves_with_adc_project_and_defaults(self, monkeypatch: pytest.MonkeyPatch, private_key):
        import google.auth

        monkeypatch.setattr(google.auth, "default", lambda: (object(), "adc-project"))
        fetched_with = self._fake_fetch(monkeypatch, private_key)
        self._set_short_form_env(monkeypatch, "gcs-log-encryption")

        loaded, error = try_load_encryption_config_from_env()

        expected = (
            "projects/adc-project/locations/global/keyRings/gcs-logs/cryptoKeys/gcs-log-encryption/cryptoKeyVersions/1"
        )
        assert error is None
        assert fetched_with == [expected]
        assert loaded is not None
        assert loaded.kms_key.project == "adc-project"
        assert loaded.kms_key.location == "global"
        assert loaded.kms_key.key_ring == "gcs-logs"
        assert loaded.kms_key.key == "gcs-log-encryption"
        assert loaded.kms_key.key_version == "1"
        assert loaded.public_key.public_numbers() == private_key.public_key().public_numbers()

    def test_short_key_name_honors_location_ring_version_overrides(self, monkeypatch: pytest.MonkeyPatch, private_key):
        import google.auth

        monkeypatch.setattr(google.auth, "default", lambda: (object(), "adc-project"))
        fetched_with = self._fake_fetch(monkeypatch, private_key)
        self._set_short_form_env(monkeypatch, "gcs-log-encryption")
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_LOCATION, "asia-south1")
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RING, "prod-ring")
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_VERSION, "7")

        loaded, error = try_load_encryption_config_from_env()

        expected = (
            "projects/adc-project/locations/asia-south1/keyRings/prod-ring"
            "/cryptoKeys/gcs-log-encryption/cryptoKeyVersions/7"
        )
        assert error is None
        assert fetched_with == [expected]
        assert loaded is not None
        assert loaded.kms_key.key_version == "7"

    def test_full_resource_wins_over_short_key_name(self, monkeypatch: pytest.MonkeyPatch, private_key):
        import google.auth

        def fail_default():
            raise AssertionError("ADC project lookup must not run when full resource is set")

        monkeypatch.setattr(google.auth, "default", fail_default)
        fetched_with = self._fake_fetch(monkeypatch, private_key)
        self._set_short_form_env(monkeypatch, "gcs-log-encryption")
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)

        loaded, error = try_load_encryption_config_from_env()

        assert error is None
        assert fetched_with == [KMS_RESOURCE]
        assert loaded is not None
        assert loaded.kms_key.project == "test-project"

    def test_error_when_adc_has_no_credentials(self, monkeypatch: pytest.MonkeyPatch):
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        def fail_default():
            raise DefaultCredentialsError("no workload identity")

        monkeypatch.setattr(google.auth, "default", fail_default)
        self._set_short_form_env(monkeypatch, "gcs-log-encryption")

        loaded, error = try_load_encryption_config_from_env()

        assert loaded is None
        assert error is not None
        assert "no workload identity" in error
        assert GCS_LOG_ENCRYPTION_KMS_KEY in error

    def test_error_when_adc_reports_no_project(self, monkeypatch: pytest.MonkeyPatch, private_key):
        import google.auth

        monkeypatch.setattr(google.auth, "default", lambda: (object(), None))
        self._fake_fetch(monkeypatch, private_key)
        self._set_short_form_env(monkeypatch, "gcs-log-encryption")

        loaded, error = try_load_encryption_config_from_env()

        assert loaded is None
        assert error is not None
        assert "did not report a GCP project" in error

    def test_short_key_name_rejects_resource_paths(self, monkeypatch: pytest.MonkeyPatch):
        self._set_short_form_env(monkeypatch, "projects/p/locations/l/keyRings/r/cryptoKeys/k/cryptoKeyVersions/1")

        loaded, error = try_load_encryption_config_from_env()

        assert loaded is None
        assert error is not None
        assert "bare KMS key name" in error


class TestEncryptLogPayload:
    def test_envelope_matches_documented_format(self, config: LogEncryptionConfig):
        envelope = json.loads(encrypt_log_payload('{"x": 1}', config, object_name="a/b.json"))
        assert set(envelope) == {"kms", "original_filename", "metadata", "encryption", "wrapped_key", "data"}
        assert envelope["metadata"] == {}
        assert envelope["kms"] == {
            "project": "test-project",
            "location": "asia-south1",
            "key_ring": "test-ring",
            "key": "test-key",
            "key_version": "1",
        }
        assert envelope["original_filename"] == "a/b.json"
        assert envelope["encryption"]["algorithm"] == "AES-256-GCM"
        assert envelope["wrapped_key"]["algorithm"] == "RSA-OAEP-2048-SHA256"
        assert len(base64.b64decode(envelope["encryption"]["nonce"])) == 12
        assert len(base64.b64decode(envelope["wrapped_key"]["ciphertext"])) == 256
        assert envelope["data"]["encoding"] == "base64"

    def test_round_trip_restores_exact_payload(self, config: LogEncryptionConfig, private_key):
        payload = json.dumps({"marker": PLAINTEXT_MARKER, "nested": {"a": [1, 2, 3]}})
        envelope = json.loads(encrypt_log_payload(payload, config, object_name="x.json"))
        assert _decrypt_envelope(envelope, private_key).decode("utf-8") == payload

    def test_metadata_stays_outside_ciphertext_and_preserves_payload(self, config: LogEncryptionConfig, private_key):
        metadata = {"log_type": "success", "correlation_id": "corr-42", "user": "jane_doe", "date": "2026-01-01"}
        payload = json.dumps({"marker": PLAINTEXT_MARKER})
        envelope = json.loads(encrypt_log_payload(payload, config, object_name="x.json", metadata=metadata))
        assert envelope["metadata"] == metadata
        assert _decrypt_envelope(envelope, private_key).decode("utf-8") == payload
        assert "corr-42" not in envelope["data"]["ciphertext"]

    def test_ciphertext_contains_no_plaintext(self, config: LogEncryptionConfig):
        envelope_str = encrypt_log_payload(PLAINTEXT_MARKER, config, object_name="x.json")
        assert PLAINTEXT_MARKER not in envelope_str
        ciphertext = base64.b64decode(json.loads(envelope_str)["data"]["ciphertext"])
        assert PLAINTEXT_MARKER.encode() not in ciphertext

    def test_each_encryption_uses_fresh_key_and_nonce(self, config: LogEncryptionConfig):
        first = json.loads(encrypt_log_payload("same", config, object_name="x.json"))
        second = json.loads(encrypt_log_payload("same", config, object_name="x.json"))
        assert first["encryption"]["nonce"] != second["encryption"]["nonce"]
        assert first["wrapped_key"]["ciphertext"] != second["wrapped_key"]["ciphertext"]
        assert first["data"]["ciphertext"] != second["data"]["ciphertext"]

    def test_tampered_ciphertext_fails_authentication(self, config: LogEncryptionConfig, private_key):
        envelope = json.loads(encrypt_log_payload("payload", config, object_name="x.json"))
        tampered = bytearray(base64.b64decode(envelope["data"]["ciphertext"]))
        tampered[0] ^= 0x01
        envelope["data"]["ciphertext"] = base64.b64encode(bytes(tampered)).decode("ascii")
        with pytest.raises(InvalidTag):
            _decrypt_envelope(envelope, private_key)


class TestEncryptedObjectName:
    def test_json_suffix_becomes_enc_json(self):
        assert encrypted_object_name("success/date=2026-01-01/a.json") == "success/date=2026-01-01/a.enc.json"

    def test_name_without_json_suffix_gets_suffix_appended(self):
        assert encrypted_object_name("a/b") == "a/b.enc.json"


class TestGcsFilenameUserPart:
    def test_email_local_part_is_sanitized(self):
        assert _gcs_filename_user_part("shaik.s@example.com") == "shaik_s"

    def test_missing_or_empty_email_is_unknown(self):
        assert _gcs_filename_user_part(None) == "unknown"
        assert _gcs_filename_user_part("") == "unknown"

    def test_special_characters_collapse_to_underscores(self):
        assert _gcs_filename_user_part("a+b*c@example.com") == "a_b_c"

    def test_local_part_with_only_special_characters_is_unknown(self):
        assert _gcs_filename_user_part("...@example.com") == "unknown"


class TestGcsLoggerEncryptionWiring:
    async def test_upload_writes_encrypted_envelope_with_user_in_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, private_key
    ):
        logger = _make_logger(monkeypatch, tmp_path, private_key)
        log_double = _GCSBaseDouble()
        logger.gcs_base = log_double

        data = {"user": {"email": "jane.doe@example.com"}, "correlation_id": "corr-99", "marker": PLAINTEXT_MARKER}
        await logger._upload_to_gcs_async(data, "bucket", "success")

        name, payload = _uploaded(log_double)
        assert name.startswith("success/date=")
        assert name.endswith("_jane_doe_corr-99.enc.json")
        assert PLAINTEXT_MARKER not in payload
        envelope = json.loads(payload)
        assert envelope["original_filename"] == name[: -len(".enc.json")] + ".json"
        metadata = envelope["metadata"]
        assert metadata["log_type"] == "success"
        assert metadata["correlation_id"] == "corr-99"
        assert metadata["user"] == "jane_doe"
        assert f"date={metadata['date']}/" in name
        assert name == f"success/date={metadata['date']}/{metadata['timestamp']}_jane_doe_corr-99.enc.json"
        decrypted = json.loads(_decrypt_envelope(envelope, private_key))
        assert decrypted == data

    async def test_upload_stays_plaintext_when_encryption_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, raising=False)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, raising=False)
        monkeypatch.delenv(GCS_LOG_ENCRYPTION_KMS_KEY, raising=False)
        logger = ProductionGCSLogger()
        log_double = _GCSBaseDouble()
        logger.gcs_base = log_double

        data = {"correlation_id": "corr-1", "marker": PLAINTEXT_MARKER}
        await logger._upload_to_gcs_async(data, "bucket", "failure")

        name, payload = _uploaded(log_double)
        assert name.startswith("failure/date=")
        assert name.endswith("_unknown_corr-1.json")
        assert not name.endswith(".enc.json")
        assert json.loads(payload) == data

    async def test_misconfigured_encryption_blocks_upload(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_PUBLIC_KEY_PATH, str(tmp_path / "missing.pem"))
        monkeypatch.setenv(GCS_LOG_ENCRYPTION_KMS_KEY_RESOURCE, KMS_RESOURCE)
        logger = ProductionGCSLogger()
        log_double = _GCSBaseDouble()
        logger.gcs_base = log_double

        await logger._upload_to_gcs_async({"marker": PLAINTEXT_MARKER}, "bucket", "success")

        assert log_double.calls == []
