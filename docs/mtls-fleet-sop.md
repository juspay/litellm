# mTLS for GPU Inference Fleets — Standard Operating Procedure

**Status: Verified in production staging on neysa-3 (vLLM) and rtx-6000-pro-blackwell-2 (SGLang + wrapper).**

Covers three deployment patterns. Pick the one matching each model's stack.

---

## 0. Shared Foundation (do once per organization)

### 0.1 One Certificate Authority for the whole fleet

```bash
# On an offline ops machine (laptop, bastion — NOT a GPU node)
mkdir -p ~/gpu-ca && cd ~/gpu-ca

openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.crt \
  -days 3650 -nodes -subj "/CN=gpu-fleet-ca" \
  -addext "basicConstraints=critical,CA:TRUE"

chmod 600 ca.key
```

- `ca.key` = master signing key. Lives ONLY here. Used ~30 seconds per new node/client. Never touches GPU nodes.
- `ca.crt` = public. Distributed to every node and every client.

### 0.2 Per-node server certificate (serverAuth profile — run per GPU node)

> **EKU discipline.** A certificate intended for use as a TLS *server* MUST be signed with `extendedKeyUsage=serverAuth` (plus `CA:FALSE` and `keyUsage`). Otherwise, because both server and client certificates share the same CA, the holder of any server key could present it as a *client* to other fleet endpoints. The engine `ssl_cert_reqs` flag does NOT check EKU, so the EKU restriction protects you only against clients that respect it — for real serverAuth/clientAuth enforcement use a reverse proxy in front or a custom SSLContext.

```bash
# On ops machine, for each node (substitute NODE_NAME, IPs)
N=neysa-3; IP_VPC=10.7.21.13; IP_PUB=103.42.51.195

openssl req -newkey rsa:2048 -keyout server-$N.key -out server-$N.csr -nodes \
  -subj "/CN=$N" \
  -addext "subjectAltName=IP:$IP_VPC,IP:$IP_PUB,IP:127.0.0.1"

cat > /tmp/server-$N.ext <<EOF
subjectAltName=IP:$IP_VPC,IP:$IP_PUB,IP:127.0.0.1
extendedKeyUsage=serverAuth
keyUsage=digitalSignature,keyEncipherment
basicConstraints=critical,CA:FALSE
EOF
openssl x509 -req -in server-$N.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server-$N.crt -days 825 -sha256 -extfile /tmp/server-$N.ext

# Deliver to node
scp server-$N.crt server-$N.key ca.crt user@node:~/mtls/
ssh user@node 'chmod 600 ~/mtls/server-*.key'

# Verify EKU + basicConstraints landed on the issued cert:
openssl x509 -in server-$N.crt -noout -text | grep -A1 "Extended Key Usage"
# Expected: X509v3 Extended Key Usage: TLS Web Server Authentication
openssl x509 -in server-$N.crt -noout -text | grep -A1 "Basic Constraints"
# Expected: CA:FALSE
```

**SAN rule:** every IP LiteLLM/clients will dial must be in `subjectAltName`, plus `127.0.0.1` for local health checks.

### 0.3 Client certificate for LiteLLM (clientAuth profile)

```bash
openssl req -newkey rsa:2048 -keyout client.key -out client.csr -nodes \
  -subj "/CN=litellm-client"

cat > /tmp/client.ext <<'EOF'
extendedKeyUsage=clientAuth
keyUsage=digitalSignature,keyEncipherment
basicConstraints=critical,CA:FALSE
EOF
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client.crt -days 365 -sha256 -extfile /tmp/client.ext

cat client.crt client.key > client.pem   # LiteLLM needs combined PEM
chmod 600 client.key client.pem

# Verify EKU:
openssl x509 -in client.crt -noout -text | grep -A1 "Extended Key Usage"
# Expected: X509v3 Extended Key Usage: TLS Web Client Authentication

# Deliver: client.pem (covers both crt+key), ca.crt → LiteLLM host
```

**ServerAuth and clientAuth must not be combined in the same cert.** If a certificate would ever need both roles (rare), issue separate certs. The EKU profile is the primary line of defense against a leased-node server key being reused as a client credential against sibling nodes.

### 0.4 The 4-state verification matrix (run after EVERY deployment)

> **Negative tests MUST include `--cacert <ca.crt>`** so the failure happens on the *server* (missing/invalid client cert), not on the client (unknown CA). We also assert the specific failure cause.

```bash
# A — no client cert, server trusts CA → server MUST refuse with a
#     certificate-required TLS alert (NOT a CA-verify error).
curl -v --cacert ca.crt https://HOST:PORT/v1/models 2>&1 | tee /tmp/testA.out
grep -qiE "alert certificate required|certificate required" /tmp/testA.out \
  || { echo "FAIL: request without client cert was accepted or wrong error"; false; }

# B — forged/self-signed client cert: server MUST refuse (unknown ca / bad cert)
openssl req -x509 -newkey rsa:2048 -nodes -keyout /tmp/w.key -out /tmp/w.crt -days 1 -subj "/CN=evil" 2>/dev/null
curl -v --cacert ca.crt --cert /tmp/w.crt --key /tmp/w.key https://HOST:PORT/v1/models 2>&1 | tee /tmp/testB.out
grep -qiE "unknown ca|bad certificate|certificate unknown" /tmp/testB.out \
  || { echo "FAIL: forged cert accepted or wrong error"; false; }

# C — valid client cert: TLS passes (may then 401 if app-layer key also set)
curl -s --cacert ca.crt --cert client.crt --key client.key https://HOST:PORT/v1/models

# D — full chain chat: MUST return model output
curl -s --cacert ca.crt --cert client.crt --key client.key \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"NAME","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
  https://HOST:PORT/v1/chat/completions
```

Expected: A=certificate-required TLS alert · B=TLS alert unknown-ca · C=JSON · D=completion JSON.

---

## 1. Pattern A — vLLM, native mTLS (NO code changes)

**Use when:** backend container runs `vllm serve ...` directly (GPUStack-style bare deployments).

**Verified:** GLM-5.3-FP8, vLLM `0.23.1rc1`, neysa-3. Works because `vllm/entrypoints/openai/cli_args.py` exposes `ssl_cert_reqs` and `ssl_ca_certs` and passes both to `uvicorn.Config`.

### 1.1 Launch flags (add exactly these 4 to existing args)

```bash
docker run -d \
  --name <model>-mtls \
  -v /home/<user>/mtls:/mtls:ro \
  <original image, network, gpus, mounts unchanged> \
  vllm serve /var/lib/gpustack/cache/huggingface/<org>/<model> \
    <…all original tuning flags unchanged…> \
    --host <bind-ip> \
    --port <port> \
    --ssl-certfile=/mtls/server.crt \
    --ssl-keyfile=/mtls/server.key \
    --ssl-ca-certs=/mtls/ca.crt \
    --ssl-cert-reqs=2
```

- `--ssl-cert-reqs=2` → `ssl.CERT_REQUIRED`; rejects every caller without a CA-signed client cert.
- Preserve `--api-key=...` if your deployment already uses bearer keys — mTLS and API-key are independent layers and can stack.
- Confirm flags parsed: startup banner `non-default args` must list all four `ssl_*` entries.

### 1.2 Verify with the 4-state matrix (§0.4) against `https://<bind-ip>:<port>`

### 1.3 LiteLLM entry

```yaml
model_list:
  - model_name: glm-5.3-fp8
    litellm_params:
      model: "openai/glm-5.3-fp8"
      api_base: "https://10.7.21.13:4042/v1"
      api_key: "os.environ/GLM_API_KEY"
      ssl_verify: "/path/to/ca.crt"

litellm_settings:
  ssl_certificate: "/path/to/client.pem"
```

### 1.4 Rollback

```bash
docker stop <model>-mtls && docker rm <model>-mtls
docker start <original-container-name>
```

### 1.5 GPUStack fanout caveat

vLLM containers launched by GPUStack share a net namespace via `--network container:<pause>`. Volume-mount `/home/<user>/mtls:/mtls:ro` **directly on the vllm container** when redeploying via GPUStack's "Backend Parameters" (or recreate the GPUStack deployment with the mount). No other GPUStack-specific change needed.

---

## 2. Pattern B — SGLang, native mTLS (5-line code patch)

**Use when:** backend container runs `python3 -m sglang.launch_server ...` directly.

**Verified:** xor-26b-a4b-v1.2, sglang `nightly-dev-20260917-408d2334`, RTX. SGLang ≥ this tag exposes `--ssl-certfile/--ssl-keyfile/--ssl-ca-certs` but hard-codes client-verify OFF — we patch it back on.

### 2.1 The patch (5 edits, one file)

File inside every sglang container: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py`.

```bash
docker exec <sglang-container> cp .../http_server.py .../http_server.py.bak

# Edit 1 & 2 (granian paths, ~lines 2473 & 2652)
docker exec <sglang-container> sed -i \
  's/ssl_verify=False,  # MTls is not supported/ssl_verify=True,  # mTLS enabled/' \
  /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py
docker exec <sglang-container> sed -i \
  's/ssl_verify=False,  # No MTLS supported for now./ssl_verify=True,  # mTLS enabled/' \
  /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py

# Edit 3–5 (uvicorn.Config call sites — add ssl_cert_reqs kwarg after each ssl_keyfile_password=)
docker exec -i <sglang-container> python3 - <<'PY'
p = "/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
s = open(p).read()
s = s.replace(
  "ssl_keyfile_password=get_serving().ssl_keyfile_password,",
  "ssl_keyfile_password=get_serving().ssl_keyfile_password,\n                    ssl_cert_reqs=2,"
)
open(p, "w").write(s)
PY

# Verify: expect 2 and 5
docker exec <sglang-container> grep -c "ssl_verify=True"  .../http_server.py
docker exec <sglang-container> grep -c "ssl_cert_reqs=2"   .../http_server.py
docker exec <sglang-container> python3 -c \
  "import py_compile; py_compile.compile('.../http_server.py', doraise=True); print('SYNTAX-OK')"
```

**Make the patch survive image updates:** put the same sed lines in a tiny Dockerfile overlay —

```dockerfile
FROM lmsysorg/sglang:nightly-dev-20260917-408d2334
RUN sed -i 's/ssl_verify=False/ssl_verify=True/' \
    /sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py
RUN python3 - <<'PY'
p="/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py"
s=open(p).read().replace(
 "ssl_keyfile_password=get_serving().ssl_keyfile_password,",
 "ssl_keyfile_password=get_serving().ssl_keyfile_password,\n                    ssl_cert_reqs=2,")
open(p,"w").write(s)
PY
```

Tag it `<org>/sglang:mtls`, use that image for all future SGLang deploys.

### 2.2 Launch

```bash
docker run -d \
  --name <model>-sglang \
  --network host \
  --gpus '"device=<N>"' \
  -v /home/<user>/mtls:/mtls:ro \
  -v <model-dir>:/model \
  <patched-or-base image> \
  python3 -m sglang.launch_server \
    --model-path /model --trust-remote-code \
    --host 0.0.0.0 --port <port> \
    <your original tuning flags> \
    --ssl-certfile /mtls/server.crt \
    --ssl-keyfile  /mtls/server.key \
    --ssl-ca-certs /mtls/ca.crt
```

Note: patch supplies the CERT_REQUIRED behavior; no extra CLI flag exists.

### 2.3 Verify (4-state matrix). Confirmed outputs on RTX:

- A: `SSL certificate problem: self-signed certificate in certificate chain`
- C: `{"object":"list","data":[{"id":"/model",...}]}`
- D: `"content":"Pong! "` from `"model":"/model"`

### 2.4 Rollback

From backup inside container (`http_server.py.bak`) or simply redeploy the stock image; restart restores plain HTTP.

---

## 3. Pattern C — HTTP wrapper in front of engine (wrapper presents the client cert)

**Use when:** a business-logic wrapper (custom FastAPI/stdlib server — JEV/XN-style scoring APIs) sits between clients and the engine. The wrapper IS the mTLS client of the engine **AND** an mTLS server to the outside world.

**Verified:** `xn_base_server.py` (stdlib `urllib.request`) + patched SGLang on same RTX host, with a TLS-terminating reverse proxy in front of the wrapper.

### 3.1 Architecture

```
external client --HTTPS + client-cert--> TLS-ingress :443 (reverse proxy)
                --HTTP (localhost)--> wrapper :8001
                --HTTPS + client-cert--> engine :8443 (Pattern B)
```

> **CRITICAL.** The solution goal is "LiteLLM → GPU node calls use mutual TLS". If the public entry point is the wrapper and the wrapper listens on **plain HTTP**, protecting only the wrapper→engine hop does NOT meet that goal: traffic on the public leg remains unencrypted. You MUST TLS-terminate the wrapper's public entry.

You have two options:

| Strategy | Cost | When to use |
|---|---|---|
| **Terminate TLS at the wrapper itself** (preferred if the wrapper is a Python/uvicorn/ASGI app) | ~6 lines of code | Wrapper runs uvicorn/hypercorn/granian |
| **TLS-terminating reverse proxy in front of the wrapper** (here we use Caddy/nginx/HAProxy/socat) | One container, no app change | Wrapper is a custom stdlib HTTP server (our case) |

Sock/socat is sufficient for many environments. For pure stdlib wrappers (e.g. `xn_base_server.py`) launching a tiny reverse proxy in front is the lowest-risk option.

Minimal Caddyfile terminating mTLS in front of the wrapper:

```caddyfile
:443 {
    tls /etc/mtls/server.crt /etc/mtls/server.key {
        client_auth {
            mode                require_and_verify
            trusted_ca_cert_file /etc/mtls/ca.crt
        }
    }
    reverse_proxy 127.0.0.1:8001
}
```

…and the wrapper itself stays on `http://127.0.0.1:8001` only. The wrapper keeps using Pattern B (patched backend URL + `client.crt/key`) to talk to the engine. External callers now must present a valid client cert to the proxy — the engine-to-wrapper and wrapper-to-proxy hops are all TLS.

### 3.2 Wrapper patch (stdlib urllib; ~8 lines)

For wrappers using `urllib.request.urlopen`:

```python
# after the existing backend URL assignment:
import ssl
SSL_CA   = os.environ.get("XN_SSL_CA",   "/mtls/ca.crt")
SSL_CERT = os.environ.get("XN_SSL_CERT", "/mtls/client.crt")
SSL_KEY  = os.environ.get("XN_SSL_KEY",  "/mtls/client.key")

_ssl_ctx = None
if SGLANG.startswith("https://"):
    _ssl_ctx = ssl.create_default_context(cafile=SSL_CA)
    _ssl_ctx.load_cert_chain(SSL_CERT, SSL_KEY)

# at the backend call site, add context:
urllib.request.urlopen(req, timeout=1800, context=_ssl_ctx)
```

For wrappers using `http.client.HTTPSConnection` use the same `context=` kwarg on the connection. For `httpx`/`aiohttp` wrappers, pass `verify=SSL_CA, cert=(SSL_CERT, SSL_KEY)` (httpx) or build an `ssl.SSLContext` for aiohttp's connector.

**Apply the patch to the host's copy of the file** (e.g. `/data2/<proj>/xn_base_server.py` — the file compose mounts) so it survives container lifecycle. Never patch a `:ro` bind-mount in place — patch the host source, restart container.

### 3.3 Wrapper run config

```bash
docker run -d --name <wrapper> \
  --network host \
  -v /home/<user>/mtls:/mtls:ro \
  -v /path/to/patched_wrapper.py:/app/xn_base_server.py:ro \
  -e XN_SGLANG_URL=https://127.0.0.1:8443 \   # https, not http
  -e XN_SSL_CA=/mtls/ca.crt \
  -e XN_SSL_CERT=/mtls/client.crt \
  -e XN_SSL_KEY=/mtls/client.key \
  <…all other envs unchanged…> \
  <image> python3 -u /app/xn_base_server.py
```

Startup log must print the `[mtls] wrapper client cert loaded` marker (or equivalent).

### 3.4 Verify

1. `docker logs <wrapper>` → cert-loaded marker, no SSL/verification errors.
2. Wrapper's public endpoint (plain HTTP as before): `curl http://127.0.0.1:8001/v1/models` → engine's model list.
3. Wrapper's functional endpoint (e.g. `POST /v1/systemone`) → real scoring JSON proves engine round-trip through mTLS.

### 3.5 Multi-wrapper note

Sibling wrappers (`jev-one-*`, `xor-f7-*`, `qwen-systemone-*`) all use the same stdlib `http.client`/`urllib` pattern with `OPENJEV_SGLANG_URL`-style env vars — the identical patch applies. When backends are comma-separated multi-lane URLs, the single module-level `_ssl_ctx` covers all of them.

---

## 4. Fleet rollout checklist

| Step | Owner | Output |
|---|---|---|
| 1. Generate fleet CA + lock away `ca.key` | SecOps | `ca.crt` distributable, `ca.key` on ops machine |
| 2. Per-node server cert w/ correct SANs, deliver to `~/mtls/` | SecOps + Node owner | `server-<node>.{crt,key}` on each node |
| 3. Client cert for LiteLLM, deliver to LiteLLM host | SecOps | `client.{crt,key,pem}` |
| 4. Decide pattern per deployment: vLLM→A, SGLang→B, wrapped→C (+B/A on engine) | Infra lead | routing doc per model |
| 5. For SGLang fleets: build patched image once (`<org>/sglang:mtls`) | Infra lead | patched image in registry |
| 6. Roll out node by node, run 4-state matrix on each | Node owner | matrix pass logged |
| 7. Add LiteLLM entries per model; restart LiteLLM once | Infra lead | end-to-end call via LiteLLM |
| 8. Close legacy plaintext ports (firewall/security-group) | NetOps | only TLS ports reachable |

## 4.5 LiteLLM Integration (client-side mTLS plumbing)

The **step 7 in the checklist** is LiteLLM-side — it does not run on GPU nodes and needs separate certs and config. This is the exact setup that worked in staging (verified with GLM-5.3 on neysa-3).

#### 4.5.1 Where certs live on the LiteLLM host

```
LiteLLM host (your Mac now, pod later)
├── ~/gpu-mtls/
│   ├── ca.crt              ← CA half, world-readable
│   ├── client.crt          ← LiteLLM identity (public)
│   ├── client.key          ← LiteLLM identity (private, chmod 600)
│   └── client.pem          ← concat of crt+key (for httpx, chmod 600)
└── litellm-config.yaml     ← per-model config below
```

**Permissions:** `chmod 600` on `.key` and `.pem`. `ls -la` should show `-rw-------` on private keys.

#### 4.5.2 Combined PEM (why required)

LiteLLM loads the client certificate into the shared `ssl.SSLContext` via `SSLContext.load_cert_chain()`, which requires the certificate **and** its private key in one file:

```bash
cat ~/gpu-mtls/client.crt ~/gpu-mtls/client.key > ~/gpu-mtls/client.pem
chmod 600 ~/gpu-mtls/client.pem
```

> **Note on upstream versions.** Stock LiteLLM ≤ 1.106.x only forwarded the client cert to the `httpx` transport. Its default transport (`aiohttp`) received the CA bundle but never loaded the client certificate, so mTLS silently degraded to one-way TLS and the server aborted with `certificate required`. The patch in §4.5.7 fixes this centrally; with it, mTLS works on **both** transports and the aiohttp fast path is preserved.

#### 4.5.3 Complete LiteLLM YAML config template

Replace paths with yours, add one block per GPU backend:

```yaml
model_list:
  # vLLM example (neysa)
  - model_name: glm-5.3-fp8
    litellm_params:
      model: "openai/glm-5.3-fp8"
      api_base: "https://10.7.21.13:4042/v1"     # direct network IP
      api_key: "os.environ/GLM_API_KEY"          # any app-layer key the backend already uses
      ssl_verify: "/run/secrets/gpu-mtls/ca.crt"

  # SGLang example (RTX node)
  - model_name: xor-26b
    litellm_params:
      model: "openai//model"                     # actual model id in the backend
      api_base: "https://127.0.0.1:8443/v1"      # local tunnel endpoint, or direct IP
      api_key: "unused"
      ssl_verify: "/run/secrets/gpu-mtls/ca.crt"

litellm_settings:
  # When ssl_certificate is configured, LiteLLM builds a COMBINED trust bundle
  # (public roots from certifi + the private CA file below). This means regular
  # calls to public-cloud providers (OpenAI, Bedrock, etc.) continue to work in
  # the SAME process as calls to the mTLS-protected GPU fleet.
  ssl_verify: "/run/secrets/gpu-mtls/ca.crt"           # private CA file
  ssl_certificate: "/run/secrets/gpu-mtls/client.pem"  # client cert+key

general_settings:
  master_key: "os.environ/GRID_MASTER_KEY"       # production API-key secret
```

Both knobs are read from `litellm_settings` because the proxy applies every key there with `setattr(litellm, key, value)`, and the SSL helpers read `litellm.ssl_certificate` / `litellm.ssl_verify`.

#### 4.5.4 Environment variables (optional, not required)

YAML alone is sufficient when the LiteLLM build includes the §4.5.7 patch. The env vars remain available as deployment-level overrides:

| Variable | Meaning | YAML equivalent |
|---|---|---|
| `SSL_CERTIFICATE` | Client cert+key PEM for mTLS | `litellm_settings.ssl_certificate` |
| `SSL_CERT_FILE` | CA bundle | `litellm_settings.ssl_verify` |
| `SSL_VERIFY` | Global verify on/off | `litellm_settings.ssl_verify` |

`SSL_CERTIFICATE` takes precedence over the YAML value. In production prefer mounting the files as secrets (`/run/secrets/...`) and setting the paths in YAML — never bake certs into the image or commit them.

#### 4.5.5 Verification from the LiteLLM side

After `litellm --config ... --port 4000` starts, verify:

```bash
# 1) List models — should return all your YAML model entries
ss -tln | grep 4000
curl -s -H "Authorization: Bearer $GRID_MASTER_KEY" http://127.0.0.1:4000/v1/models | head

# 2) Test a chat completion — goes via mTLS to the backend
curl -s -H "Authorization: Bearer $GRID_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"glm-5.3-fp8","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
     http://127.0.0.1:4000/v1/chat/completions
```

**Failure modes to watch in LiteLLM logs:**

| Log line | Cause | Fix |
|---|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` (server cert) | `ssl_verify` path wrong or CA missing | Check path + SAN list on server cert |
| `certificate required` / `alert certificate required` | Client cert not loaded (unpatched build) | Apply §4.5.7 or set `SSL_CERTIFICATE` |
| `SSL_CERTIFICATE ... does not exist` warning | Path typo in YAML/env | Fix the path |
| `[Errno 13] Permission denied` | `client.pem` not readable | `chmod 600` + correct owner |
| `401 Unauthorized` | mTLS passed but `api_key` wrong | fix `api_key` in YAML |
| `Connection refused` on GPU IP | backend not listening | check backend startup args |

#### 4.5.6 Testing the two layers independently

```bash
# Direct mTLS test from LiteLLM host (bypassing LiteLLM auth):
curl -v --cacert ~/gpu-mtls/ca.crt \
     --cert ~/gpu-mtls/client.crt \
     --key ~/gpu-mtls/client.key \
     https://10.7.21.13:4042/v1/models

# vs broken (no client cert):
curl -v --cacert ~/gpu-mtls/ca.crt https://10.7.21.13:4042/v1/models
```

Proves: client cert = handshake succeeds; without it = handshake fails.

#### 4.5.7 Required code change (keep aiohttp, enable mTLS)

Only needed if the deployed LiteLLM predates this fix. The change is small and central — it does **not** disable or replace the aiohttp transport, so your existing performance profile is preserved.

**File:** `litellm/llms/custom_httpx/http_handler.py`

1. New helper resolving the client cert (env override, then YAML setting, with a missing-path warning):

```python
def get_ssl_client_certificate() -> Optional[str]:
    client_cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)
    if client_cert and not os.path.exists(client_cert):
        verbose_logger.warning(...)  # never crash startup
        return None
    return client_cert
```

2. `_create_ssl_context()` gains a `client_cert` argument and loads it:

```python
if client_cert:
    try:
        custom_ssl_context.load_cert_chain(certfile=client_cert)
    except (ssl.SSLError, OSError, ValueError) as e:
        verbose_logger.warning(...)  # degrade to one-way TLS, don't fail the request
```

3. `get_ssl_configuration()` resolves the client cert and includes it in the cache key (so two identities never share a cached context).

4. Both `AsyncHTTPHandler.create_client` and `HTTPHandler.__init__` use the helper for `cert=` (no behavior change for non-mTLS users).

**Why this works on aiohttp:** the resulting `ssl.SSLContext` is passed to the aiohttp `TCPConnector` and per-request `ssl=` kwarg, so the client certificate is presented during the handshake — with no custom connector code and no change to concurrency/limits.

**Validation performed** (local strict-mTLS server, `verify_mode=CERT_REQUIRED`):

| Case | Result |
|---|---|
| aiohttp + no client cert → mTLS server | ✅ rejected (handshake fails) |
| aiohttp + client cert → mTLS server | ✅ 200, server observed expected client CN |
| aiohttp + global client cert → plain TLS server | ✅ 200 (no regression) |
| aiohttp + no client cert → plain TLS server | ✅ 200 (no regression) |

Regression tests live in `tests/test_litellm/llms/custom_httpx/test_ssl_client_certificate.py`.

## 5. Maintenance

- **Cert rotation:** server certs 825-day, client certs 365-day. Reissue from the same CA; restart affected containers; no LiteLLM change needed (same client.pem path).
- **Revocation (what actually removes the attacker's access).** Saying "delete the client cert at LiteLLM" does NOT revoke anything — the attacker's copy is still valid under the trusted CA. To revoke, you MUST change what the *servers* will now accept. Pick one mechanism and test it end-to-end:
  1. **Per-endpoint CN allowlist (simplest & fast).** Require each engine (or the proxy in front of it) to also enforce a known-client CN allowlist (e.g. socat `cf-peer-type=cert/cn=litellm-client`). Drop the stolen CN/fingerprint from the list and restart the affected endpoints. This is the recommended approach for fleets of <20 nodes.
  2. **CRL distribution.** Generate a CRL with your CA (e.g. `openssl ca -gencrl`) and roll it out to each endpoint that checks CRLs (nginx `ssl_crl`, socat `*crl*`). Confirm a revoked cert is rejected while unrelated ones still work.
  3. **Rotate the CA.** Issue a new CA + certificates, deploy new `ca.crt`/`server.crt`/`client.pem` everywhere, restart endpoints, and remove the old CA from trust. Required if `ca.key` itself leaks.

  The "delete at LiteLLM + rotate server" pattern alone is NOT an actual revocation — the attacker's copy still authenticates. Always pair it with (1), (2), or (3) above.
- **Key compromise:** `ca.key` compromise = full fleet reissue. Keep it offline.
- **Engine upgrades:** vLLM — flags are stable, no work. SGLang — rebuild patched image from new tag, rerun 4-state matrix on one canary before fleet.
- **Adding a node:** §0.2 + appropriate pattern + matrix = ~15 min.

## 6. Known limits (deliberate scope cuts)

- mTLS secures the **LiteLLM → engine** leg and the **wrapper → engine** leg. End-user → LiteLLM remains API-key over your existing TLS termination.
- Internal RFC1918 node IPs — no public CA possible; that's exactly why a private CA is used. Do not publish `ca.crt` signed endpoints to the internet.
- llama.cpp / GGUF backends (`llama-server`) support client-side TLS only via flags for server cert — no client-cert enforcement. Wrap with socat/nginx if those models must join the mTLS fleet (Pattern: proxy, out of scope here).
