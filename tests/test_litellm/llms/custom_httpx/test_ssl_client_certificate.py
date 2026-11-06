"""
Mutual-TLS coverage for the shared SSL context used by both aiohttp and httpx.

These tests are intentionally heavier than unit stubs:
 * they exercise the actual request path through LiteLLM's transport
   (so removing ``load_cert_chain`` cannot leave all tests passing)
 * they assert the *observed* client identity on the server
 * they exercise negative cases (no cert, wrong cert) and streaming
 * they assert a configured CA file stays the only trust anchor
"""

import json
import ssl
import warnings
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import litellm
from litellm.llms.custom_httpx import http_handler
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    SSLCertificateError,
    _create_ssl_context,
    get_ssl_client_certificate,
    get_ssl_configuration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures: a real CA with proper EKUs (serverAuth for server,
# clientAuth for LiteLLM). The profile mirrors the recommended fleet SOP.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def mtls_material(tmp_path_factory):
    root = tmp_path_factory.mktemp("mtls")
    ca_key = root / "ca.key"
    ca_crt = root / "ca.crt"

    # CA with explicit CA:TRUE basic constraint.
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_crt),
            "-days",
            "2",
            "-subj",
            "/CN=test-fleet-ca",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
        ],
        check=True,
        capture_output=True,
    )

    def _issue(cn: str, dir_name: str, ext: str):
        d = root / dir_name
        d.mkdir(exist_ok=True)
        key, csr, crt = d / "key.pem", d / "csr.pem", d / "crt.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(csr),
                "-subj",
                f"/CN={cn}",
                "-addext",
                "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        (d / "ext.cnf").write_text(ext)
        subprocess.run(
            [
                "openssl",
                "x509",
                "-req",
                "-in",
                str(csr),
                "-CA",
                str(ca_crt),
                "-CAkey",
                str(ca_key),
                "-CAcreateserial",
                "-out",
                str(crt),
                "-days",
                "2",
                "-sha256",
                "-extfile",
                str(d / "ext.cnf"),
            ],
            check=True,
            capture_output=True,
        )
        return {"key": str(key), "crt": str(crt)}

    server = _issue(
        "test-server",
        "server",
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "extendedKeyUsage=serverAuth\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "basicConstraints=critical,CA:FALSE\n",
    )

    client = _issue(
        "litellm-client",
        "client",
        "extendedKeyUsage=clientAuth\nkeyUsage=digitalSignature,keyEncipherment\nbasicConstraints=critical,CA:FALSE\n",
    )
    client_pem = root / "client.pem"
    client_pem.write_text(Path(client["crt"]).read_text() + Path(client["key"]).read_text())

    return {
        "ca_crt": str(ca_crt),
        "server": server,
        "client_pem": str(client_pem),
    }


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        peer = self.connection.getpeercert()
        cn = None
        if peer:
            for item in peer.get("subject", ()):
                for pair in item:
                    if pair[0] == "commonName":
                        cn = pair[1]
        body = json.dumps({"object": "list", "client_cn": cn, "data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _start_strict_mtls_server(mtls_material, port=0):
    """A real strict mTLS server (CERT_REQUIRED) that echoes the client's CN."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(mtls_material["server"]["crt"], mtls_material["server"]["key"])
    ctx.load_verify_locations(cafile=mtls_material["ca_crt"])
    ctx.verify_mode = ssl.CERT_REQUIRED

    srv = ThreadingHTTPServer(("127.0.0.1", port), _RecordingHandler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, srv.server_address[1]


@pytest.fixture(scope="session")
def mtls_server(mtls_material):
    srv, port = _start_strict_mtls_server(mtls_material, port=0)
    yield {
        "host": "127.0.0.1",
        "port": port,
        "ca_crt": mtls_material["ca_crt"],
        "client_pem": mtls_material["client_pem"],
    }
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("SSL_CERTIFICATE", raising=False)
    original = litellm.ssl_certificate
    original_verify = litellm.ssl_verify
    http_handler._ssl_context_cache.clear()
    yield
    litellm.ssl_certificate = original
    litellm.ssl_verify = original_verify
    http_handler._ssl_context_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration resolution
# ─────────────────────────────────────────────────────────────────────────────
def test_config_client_from_env(monkeypatch, mtls_material):
    monkeypatch.setenv("SSL_CERTIFICATE", mtls_material["client_pem"])
    assert get_ssl_client_certificate() == mtls_material["client_pem"]


def test_config_client_from_litellm_setting(mtls_material):
    litellm.ssl_certificate = mtls_material["client_pem"]
    assert get_ssl_client_certificate() == mtls_material["client_pem"]


def test_config_none_when_unset():
    litellm.ssl_certificate = None
    assert get_ssl_client_certificate() is None


# ─────────────────────────────────────────────────────────────────────────────
# P2: a bad configured client identity MUST be a hard error, never a silent
# downgrade.  Crucially, the failure must be raised *every time*, not masked
# by the context cache on a subsequent request.
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_client_file_raises():
    litellm.ssl_certificate = "/nonexistent/does-not-exist.pem"
    with pytest.raises(SSLCertificateError):
        get_ssl_client_certificate()


def test_missing_client_raises_on_every_call():
    litellm.ssl_certificate = "/nonexistent/does-not-exist.pem"
    with pytest.raises(SSLCertificateError):
        _ = get_ssl_configuration(True)
    # Same path again: the first failure must NOT be cached.  Expectation is
    # a second raise, not a successful cached (degraded) context.
    with pytest.raises(SSLCertificateError):
        _ = get_ssl_configuration(True)


def test_corrupt_client_file_raises(tmp_path):
    bad = tmp_path / "broken.pem"
    bad.write_text("-----BEGIN CERTIFICATE-----\ngarbage\n-----END CERTIFICATE-----\n")
    litellm.ssl_certificate = str(bad)
    with pytest.raises(SSLCertificateError):
        _ = _create_ssl_context(
            cafile=None,
            ssl_security_level=None,
            ssl_ecdh_curve=None,
            client_cert=str(bad),
        )


@pytest.mark.parametrize("via_global", [False, True])
def test_custom_ca_file_is_the_only_trust_anchor(mtls_material, via_global):
    if via_global:
        litellm.ssl_verify = mtls_material["ca_crt"]
        ctx = get_ssl_configuration()
    else:
        ctx = get_ssl_configuration(mtls_material["ca_crt"])

    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.cert_store_stats()["x509_ca"] == 1
    assert ctx.get_ca_certs()[0]["subject"] == ((("commonName", "test-fleet-ca"),),)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end integration. Patches in http_handler are actually exercised, so
# removing ``load_cert_chain`` makes these fail, not pass.
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_real_mtls_handshake_presents_client_identity(mtls_server, monkeypatch):
    """Default aiohttp transport, valid client cert → server observes CN."""
    litellm.ssl_certificate = mtls_server["client_pem"]
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "False")

    handler = AsyncHTTPHandler(ssl_verify=mtls_server["ca_crt"])
    try:
        r = await handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
        assert r.status_code == 200
        assert r.json()["client_cn"] == "litellm-client"
    finally:
        await handler.client.aclose()


@pytest.mark.asyncio
async def test_real_mtls_without_client_cert_is_rejected(mtls_server, monkeypatch):
    """Same strict server, no client cert → handshake must fail before HTTP."""
    litellm.ssl_certificate = None
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "False")

    handler = AsyncHTTPHandler(ssl_verify=mtls_server["ca_crt"])
    with pytest.raises(Exception) as exc:
        await handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
    msg = str(exc.value).lower()
    # Handshake-level missing-cert rejection, not a bare connection timeout
    # (which could mask the absence of an entrypoint listener).
    assert any(w in msg for w in ("certificate", "handshake")), msg
    await handler.client.aclose()


@pytest.mark.asyncio
async def test_real_mtls_with_wrong_ca_is_rejected(mtls_server, monkeypatch, tmp_path):
    """Client cert signed by a *different* CA is refused."""
    # Build a rogue self-signed cert with compatible extensions.
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tmp_path / "wrong.key"),
            "-out",
            str(tmp_path / "wrong.crt"),
            "-days",
            "1",
            "-subj",
            "/CN=rogue-client",
            "-addext",
            "extendedKeyUsage=clientAuth",
            "-addext",
            "basicConstraints=critical,CA:FALSE",
        ],
        capture_output=True,
        check=True,
    )
    pem = tmp_path / "wrong.pem"
    pem.write_text((tmp_path / "wrong.crt").read_text() + (tmp_path / "wrong.key").read_text())
    litellm.ssl_certificate = str(pem)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "False")

    handler = AsyncHTTPHandler(ssl_verify=mtls_server["ca_crt"])
    with pytest.raises(Exception) as exc:
        await handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
    text = str(exc.value)
    assert any(t in text for t in ("tlsv1 alert", "tlsv13 alert", "certificate", "unknown ca")), text
    await handler.client.aclose()


@pytest.mark.asyncio
async def test_real_mtls_streaming_path(mtls_server, monkeypatch):
    """streaming=True goes through the same context - covers llm_http_handler
    streaming path that previously used a bare AsyncHTTPHandler without an
    injected client identity on the aiohttp connector.
    """
    litellm.ssl_certificate = mtls_server["client_pem"]
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "False")

    handler = AsyncHTTPHandler(ssl_verify=mtls_server["ca_crt"])
    try:
        async with handler.client.stream("GET", f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models") as r:
            assert r.status_code == 200
            body = await r.aread()
            assert json.loads(body)["client_cn"] == "litellm-client"
    finally:
        await handler.client.aclose()


def test_client_cert_cache_is_keyed_on_cert(mtls_material):
    """Two different configured identities must never share a cached context."""
    litellm.ssl_certificate = None
    first = get_ssl_configuration(True)
    litellm.ssl_certificate = mtls_material["client_pem"]
    second = get_ssl_configuration(True)
    assert first is not second


def test_failed_client_cert_load_is_not_cached(mtls_material, tmp_path):
    pem = tmp_path / "client.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\ngarbage\n-----END CERTIFICATE-----\n")
    litellm.ssl_certificate = str(pem)
    with pytest.raises(SSLCertificateError):
        get_ssl_configuration(mtls_material["ca_crt"])

    pem.write_text(Path(mtls_material["client_pem"]).read_text())
    ctx = get_ssl_configuration(mtls_material["ca_crt"])
    assert isinstance(ctx, ssl.SSLContext)


def test_real_mtls_sync_handler_presents_client_identity(mtls_server):
    litellm.ssl_certificate = mtls_server["client_pem"]

    handler = HTTPHandler(ssl_verify=mtls_server["ca_crt"])
    try:
        r = handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
        assert r.status_code == 200
        assert r.json()["client_cn"] == "litellm-client"
    finally:
        handler.close()


@pytest.mark.asyncio
async def test_handlers_do_not_reload_cert_into_shared_context(mtls_server):
    litellm.ssl_certificate = mtls_server["client_pem"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sync_handler = HTTPHandler(ssl_verify=mtls_server["ca_crt"])
        async_handler = AsyncHTTPHandler(ssl_verify=mtls_server["ca_crt"])
    sync_handler.close()
    await async_handler.client.aclose()

    assert not [w for w in caught if "`cert=...` is deprecated" in str(w.message)]


def test_caller_supplied_context_presents_client_identity_sync(mtls_server):
    litellm.ssl_certificate = mtls_server["client_pem"]
    caller_ctx = ssl.create_default_context(cafile=mtls_server["ca_crt"])

    handler = HTTPHandler(ssl_verify=caller_ctx)
    try:
        r = handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
        assert r.status_code == 200
        assert r.json()["client_cn"] == "litellm-client"
    finally:
        handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_aiohttp", ["False", "True"])
async def test_caller_supplied_context_presents_client_identity_async(mtls_server, monkeypatch, disable_aiohttp):
    litellm.ssl_certificate = mtls_server["client_pem"]
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", disable_aiohttp)
    caller_ctx = ssl.create_default_context(cafile=mtls_server["ca_crt"])

    handler = AsyncHTTPHandler(ssl_verify=caller_ctx)
    try:
        r = await handler.client.get(f"https://{mtls_server['host']}:{mtls_server['port']}/v1/models")
        assert r.status_code == 200
        assert r.json()["client_cn"] == "litellm-client"
    finally:
        await handler.client.aclose()
