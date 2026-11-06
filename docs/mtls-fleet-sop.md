# mTLS for GPU Inference Fleets: Standard Operating Procedure

**Status.** The engine flags, SGLang patch and wrapper pattern were verified in staging on neysa-3 (vLLM) and rtx-6000-pro-blackwell-2 (SGLang + wrapper) with a single shared CA. The split server/client CA layout below (§0.1) and the revocation procedure (§5.2) were verified locally against uvicorn with the same `--ssl-ca-certs` / `--ssl-cert-reqs 2` settings vLLM passes through; re-run §0.4 on each node when migrating it to the split layout.

Covers three deployment patterns. Pick the one matching each model's stack.

---

## 0. Shared Foundation (do once per organization)

### 0.1 Two CAs: one for servers, one for clients

```bash
# On a locked-down ops machine (bastion or SecOps laptop; NOT a GPU node or the LiteLLM host)
mkdir -p ~/gpu-ca && cd ~/gpu-ca

# Signs GPU node server certificates. LiteLLM and wrappers trust it (ssl_verify / --cacert).
openssl req -x509 -newkey rsa:4096 -keyout server-ca.key -out server-ca.crt \
  -days 3650 -nodes -subj "/CN=gpu-fleet-server-ca" \
  -addext "basicConstraints=critical,CA:TRUE"

# Signs client certificates (LiteLLM, wrappers). GPU nodes trust it (--ssl-ca-certs).
openssl req -x509 -newkey rsa:4096 -keyout client-ca.key -out client-ca.crt \
  -days 3650 -nodes -subj "/CN=gpu-fleet-client-ca-v1" \
  -addext "basicConstraints=critical,CA:TRUE"

chmod 600 server-ca.key client-ca.key
```

- `server-ca.key` and `client-ca.key` live ONLY here and never touch GPU nodes.
- `server-ca.crt` goes to every client (LiteLLM host, wrappers) and to every node (for the §0.4 case E check). `client-ca.crt` goes to every node.

**Why two CAs.** None of the engines can check a CRL or a client CN allowlist, so the only thing a node can be told to stop accepting is an issuing CA. Keeping clients on their own CA means revoking a client certificate (§5.2) only rotates `client-ca` and the handful of client certs, never the server certs. It also means a stolen node server key can never authenticate as a client: nodes do not trust its issuer for clients at all, independent of EKU.

**Migrating from the single shared `ca.crt`.** Rename the existing `ca.crt`/`ca.key` to `server-ca.crt`/`server-ca.key`; existing server certs stay valid and LiteLLM's `ssl_verify` keeps working. Create `client-ca` as above, issue new client certs from it (§0.3), then switch each node's `--ssl-ca-certs` to `client-ca.crt` and roll LiteLLM's `client.pem` at the same time.

### 0.2 Per-node server certificate (serverAuth profile; run per GPU node)

> **EKU discipline.** Server certificates MUST carry `extendedKeyUsage=serverAuth` (plus `CA:FALSE` and `keyUsage`). TLS servers check the peer certificate's purpose during chain verification, not just its issuer. OpenSSL (used by uvicorn under vLLM and SGLang, and by nginx) applies the `sslclient` purpose when verifying a client, so a serverAuth-only certificate presented as a client is rejected with an `unsupported certificate` alert; Caddy's Go `crypto/tls` likewise requires the clientAuth EKU on client certificates. No extra configuration is needed. Matrix case E (§0.4) proves it on every node. With the split CA (§0.1) this is the second line of defense; the first is that nodes do not trust `server-ca` for clients.

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
openssl x509 -req -in server-$N.csr -CA server-ca.crt -CAkey server-ca.key -CAcreateserial \
  -out server-$N.crt -days 825 -sha256 -extfile /tmp/server-$N.ext

# Deliver to node: its own server cert/key, the CA it uses to verify CLIENTS, and server-ca.crt for §0.4 case E
scp server-$N.crt server-$N.key client-ca.crt server-ca.crt user@node:~/mtls/
ssh user@node "cd ~/mtls && mv server-$N.crt server.crt && mv server-$N.key server.key && chmod 600 server.key"

# Verify EKU + basicConstraints landed on the issued cert:
openssl x509 -in server-$N.crt -noout -text | grep -A1 "Extended Key Usage"
# Expected: X509v3 Extended Key Usage: TLS Web Server Authentication
openssl x509 -in server-$N.crt -noout -text | grep -A1 "Basic Constraints"
# Expected: CA:FALSE
```

**SAN rule:** every IP LiteLLM/clients will dial must be in `subjectAltName`, plus `127.0.0.1` for local health checks.

### 0.3 Client certificate for LiteLLM (clientAuth profile)

Give every issued client certificate a unique CN (`litellm-<env>-<YYYYMMDDHHMM>`, minute resolution so a same-day reissue still differs) so node logs and §0.4 output show exactly which credential connected. Revocation does not rely on the CN (§5.2), but you cannot investigate a leak without it.

```bash
C=litellm-staging-$(date +%Y%m%d%H%M)

openssl req -newkey rsa:2048 -keyout $C.key -out $C.csr -nodes -subj "/CN=$C"

cat > /tmp/client.ext <<'EOF'
extendedKeyUsage=clientAuth
keyUsage=digitalSignature,keyEncipherment
basicConstraints=critical,CA:FALSE
EOF
openssl x509 -req -in $C.csr -CA client-ca.crt -CAkey client-ca.key -CAcreateserial \
  -out $C.crt -days 365 -sha256 -extfile /tmp/client.ext

cat $C.crt $C.key > client.pem   # LiteLLM needs cert + key in one PEM
cp $C.crt client.crt && cp $C.key client.key
chmod 600 client.key client.pem $C.key

# Verify EKU:
openssl x509 -in client.crt -noout -text | grep -A1 "Extended Key Usage"
# Expected: X509v3 Extended Key Usage: TLS Web Client Authentication

# Deliver client.pem and server-ca.crt to the LiteLLM host
```

**ServerAuth and clientAuth must not be combined in the same cert.** If a host needs both roles (a wrapper in Pattern C), issue it one server cert from `server-ca` and one client cert from `client-ca`.

### 0.4 The verification matrix (run after EVERY deployment)

All cases use the same `--cacert server-ca.crt`. Case C runs first: once it gets an HTTP response, the client side's trust of the server is proven, so a later failure that differs only in the presented client certificate is the *server* refusing the client. Each refusal is also checked by curl's exit status: 35, 52, 55 or 56 mean the TLS exchange was aborted by the server; 60 would mean curl rejected the server's certificate, and 7 or 28 that nothing is listening, so those count as failures of the test, not passes.

Run A, B, C and D from the LiteLLM host (it has `client.pem` and `server-ca.crt`). Run E on the node itself, from `~/mtls`, so the server key never leaves the node.

```bash
HOST=10.7.21.13; PORT=4042      # node under test (same values on the LiteLLM host and on the node)
probe() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' --cacert server-ca.crt "$@" "https://$HOST:$PORT/v1/models" 2>/dev/null)
  echo "$code $?"
}
server_refused() { case "$1" in "000 35"|"000 52"|"000 55"|"000 56") return 0 ;; esac; return 1; }

# C: valid client cert. TLS must pass (401 is fine if an app-layer key is also enforced).
r=$(probe --cert client.pem); [ "${r%% *}" != 000 ] || echo "FAIL C: valid client cert got no HTTP response ($r)"

# A: no client cert. Server must refuse during the handshake.
r=$(probe); server_refused "$r" || echo "FAIL A: no-cert request was not refused by the server ($r)"

# B: forged self-signed client cert. Server must refuse.
openssl req -x509 -newkey rsa:2048 -nodes -keyout /tmp/w.key -out /tmp/w.crt -days 1 -subj "/CN=evil" \
  -addext "extendedKeyUsage=clientAuth" 2>/dev/null
r=$(probe --cert /tmp/w.crt --key /tmp/w.key); server_refused "$r" || echo "FAIL B: forged cert was not refused by the server ($r)"

# D: full chain chat. Must return model output.
curl -s --cacert server-ca.crt --cert client.pem \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"NAME","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
  "https://$HOST:$PORT/v1/chat/completions"
```

```bash
# E, on the node (cd ~/mtls; define HOST, PORT, probe and server_refused as above):
# the node's own server cert presented as a client. Server must refuse.
r=$(probe --cert server.crt --key server.key); server_refused "$r" || echo "FAIL E: server cert was not refused as a client ($r)"
```

Expected: no `FAIL` lines, and D prints a completion JSON. With `curl -v`, OpenSSL builds of curl usually also name the server's alert (`certificate required` for A, `unknown ca` for B and E); some TLS stacks only report a reset or empty reply, which is why the exit status, not the alert text, is the pass criterion.

---

## 1. Pattern A: vLLM, native mTLS (NO code changes)

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
    --ssl-ca-certs=/mtls/client-ca.crt \
    --ssl-cert-reqs=2
```

- `--ssl-cert-reqs=2` is `ssl.CERT_REQUIRED`; rejects every caller without a client cert issued by `client-ca`.
- `--ssl-ca-certs` is the CA the node trusts for *clients*. It is also the revocation lever (§5.2).
- Preserve `--api-key=...` if your deployment already uses bearer keys; mTLS and API-key are independent layers and can stack.
- Confirm flags parsed: startup banner `non-default args` must list all four `ssl_*` entries.

### 1.2 Verify with the matrix (§0.4) against `https://<bind-ip>:<port>`

### 1.3 LiteLLM entry

See §4.5.3 for the full template and for why `ssl_verify` is set under `litellm_settings`.

```yaml
model_list:
  - model_name: glm-5.3-fp8
    litellm_params:
      model: "openai/glm-5.3-fp8"
      api_base: "https://10.7.21.13:4042/v1"
      api_key: "os.environ/GLM_API_KEY"

litellm_settings:
  ssl_verify: "/path/to/server-ca.crt"
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

## 2. Pattern B: SGLang, native mTLS (5-line code patch)

**Use when:** backend container runs `python3 -m sglang.launch_server ...` directly.

**Verified:** xor-26b-a4b-v1.2, sglang `nightly-dev-20260917-408d2334`, RTX. SGLang at or after this tag exposes `--ssl-certfile/--ssl-keyfile/--ssl-ca-certs` but hard-codes client-verify OFF; we patch it back on.

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

# Edit 3–5 (uvicorn.Config call sites; add ssl_cert_reqs kwarg after each ssl_keyfile_password=)
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

**Make the patch survive image updates:** put the same sed lines in a tiny Dockerfile overlay:

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
    --ssl-ca-certs /mtls/client-ca.crt
```

Note: patch supplies the CERT_REQUIRED behavior; no extra CLI flag exists.

### 2.3 Verify with the matrix (§0.4). Confirmed on RTX:

- C: `{"object":"list","data":[{"id":"/model",...}]}`
- D: `"content":"Pong! "` from `"model":"/model"`

### 2.4 Rollback

From backup inside container (`http_server.py.bak`) or simply redeploy the stock image; restart restores plain HTTP.

---

## 3. Pattern C: HTTP wrapper in front of engine (wrapper presents the client cert)

**Use when:** a business-logic wrapper (custom FastAPI/stdlib server, e.g. JEV/XN-style scoring APIs) sits between clients and the engine. The wrapper IS the mTLS client of the engine **AND** sits behind an mTLS server facing the outside world.

**Verified:** `xn_base_server.py` (stdlib `urllib.request`) + patched SGLang on same RTX host, with a TLS-terminating reverse proxy in front of the wrapper.

### 3.1 Architecture

```
external client --HTTPS + client-cert--> TLS-ingress :443 (reverse proxy)
                --HTTP (localhost)--> wrapper :8001
                --HTTPS + client-cert--> engine :8443 (Pattern B)
```

> **CRITICAL.** The solution goal is "LiteLLM to GPU node calls use mutual TLS". If the public entry point is the wrapper and the wrapper listens on **plain HTTP**, protecting only the wrapper-to-engine hop does NOT meet that goal: traffic on the public leg remains unencrypted. You MUST TLS-terminate the wrapper's public entry.

Either way the public entry must *require* a client certificate from `client-ca`; one-way TLS there is not enough. Two options:

| Strategy | Cost | When to use |
|---|---|---|
| **Terminate mTLS at the wrapper itself** (preferred if the wrapper is a Python/uvicorn/ASGI app) | launch flags only | Wrapper runs uvicorn: the same four `--ssl-*` flags as Pattern A, with `--ssl-ca-certs client-ca.crt --ssl-cert-reqs 2` |
| **mTLS-terminating reverse proxy in front of the wrapper** (Caddy/nginx/HAProxy) | One container, no app change | Wrapper is a custom stdlib HTTP server (our case) |

Minimal Caddyfile terminating mTLS in front of the wrapper. The ingress presents the node's server cert and trusts `client-ca` for callers, exactly like an engine in Pattern A/B:

```caddyfile
:443 {
    tls /etc/mtls/server.crt /etc/mtls/server.key {
        client_auth {
            mode                require_and_verify
            trusted_ca_cert_file /etc/mtls/client-ca.crt
        }
    }
    reverse_proxy 127.0.0.1:8001
}
```

The wrapper itself stays on `http://127.0.0.1:8001` only, and talks to the engine with its **own** client cert (issued from `client-ca` per §0.3, CN `wrapper-<name>-<YYYYMMDDHHMM>`), trusting `server-ca` for the engine's cert.

### 3.2 Wrapper patch (stdlib urllib; ~8 lines)

For wrappers using `urllib.request.urlopen`:

```python
# after the existing backend URL assignment:
import ssl
SSL_CA   = os.environ.get("XN_SSL_CA",   "/mtls/server-ca.crt")
SSL_CERT = os.environ.get("XN_SSL_CERT", "/mtls/client.crt")
SSL_KEY  = os.environ.get("XN_SSL_KEY",  "/mtls/client.key")

_ssl_ctx = None
if SGLANG.startswith("https://"):
    _ssl_ctx = ssl.create_default_context(cafile=SSL_CA)
    _ssl_ctx.load_cert_chain(SSL_CERT, SSL_KEY)

# at the backend call site, add context:
urllib.request.urlopen(req, timeout=1800, context=_ssl_ctx)
```

For wrappers using `http.client.HTTPSConnection` use the same `context=` kwarg on the connection. For `httpx`/`aiohttp` wrappers, build the same `ssl.SSLContext` and pass it as `verify=` (httpx) or as the connector's `ssl=` (aiohttp).

**Apply the patch to the host's copy of the file** (e.g. `/data2/<proj>/xn_base_server.py`, the file compose mounts) so it survives container lifecycle. Never patch a `:ro` bind-mount in place; patch the host source, restart container.

### 3.3 Wrapper run config

The engine URL must be `https://`, not `http://`.

```bash
docker run -d --name <wrapper> \
  --network host \
  -v /home/<user>/mtls:/mtls:ro \
  -v /path/to/patched_wrapper.py:/app/xn_base_server.py:ro \
  -e XN_SGLANG_URL=https://127.0.0.1:8443 \
  -e XN_SSL_CA=/mtls/server-ca.crt \
  -e XN_SSL_CERT=/mtls/client.crt \
  -e XN_SSL_KEY=/mtls/client.key \
  <…all other envs unchanged…> \
  <image> python3 -u /app/xn_base_server.py
```

Startup log must print the `[mtls] wrapper client cert loaded` marker (or equivalent).

### 3.4 Verify

1. `docker logs <wrapper>`: cert-loaded marker, no SSL/verification errors.
2. Run the §0.4 matrix against the ingress (`https://<node-ip>:443`), not against `:8001`.
3. Wrapper's functional endpoint (e.g. `POST /v1/systemone`) through the ingress with the client cert returns real scoring JSON, proving the engine round-trip through mTLS.

### 3.5 Multi-wrapper note

Sibling wrappers (`jev-one-*`, `xor-f7-*`, `qwen-systemone-*`) all use the same stdlib `http.client`/`urllib` pattern with `OPENJEV_SGLANG_URL`-style env vars; the identical patch applies. When backends are comma-separated multi-lane URLs, the single module-level `_ssl_ctx` covers all of them.

---

## 4. Fleet rollout checklist

| Step | Owner | Output |
|---|---|---|
| 1. Generate `server-ca` and `client-ca`, lock away both keys | SecOps | `server-ca.crt`, `client-ca.crt` distributable; keys on ops machine |
| 2. Per-node server cert w/ correct SANs, deliver with `client-ca.crt` to `~/mtls/` | SecOps + Node owner | `server.{crt,key}`, `client-ca.crt` on each node |
| 3. Client cert for LiteLLM (unique CN), deliver with `server-ca.crt` to LiteLLM host | SecOps | `client.pem`, `server-ca.crt` |
| 4. Decide pattern per deployment: vLLM to A, SGLang to B, wrapped to C (+B/A on engine) | Infra lead | routing doc per model |
| 5. For SGLang fleets: build patched image once (`<org>/sglang:mtls`) | Infra lead | patched image in registry |
| 6. Roll out node by node, run §0.4 matrix on each | Node owner | matrix pass logged |
| 7. Add LiteLLM entries per model; restart LiteLLM once | Infra lead | end-to-end call via LiteLLM |
| 8. Close legacy plaintext ports (firewall/security-group) | NetOps | only TLS ports reachable |
| 9. Rehearse §5.2 revocation once on a canary node | SecOps | stolen-cert-fails / replacement-works output logged |

## 4.5 LiteLLM Integration (client-side mTLS plumbing)

**Step 7 in the checklist** is LiteLLM-side. It does not run on GPU nodes and needs separate certs and config. Verified in staging with GLM-5.3 on neysa-3.

#### 4.5.1 Where certs live on the LiteLLM host

```
LiteLLM host
├── /run/secrets/gpu-mtls/
│   ├── server-ca.crt       ← verifies GPU node server certs
│   ├── trust-bundle.pem    ← only if LiteLLM also calls public providers (§4.5.3)
│   └── client.pem          ← LiteLLM identity, cert + key (chmod 600)
└── litellm-config.yaml
```

**Permissions:** `chmod 600` on `client.pem`. `ls -la` should show `-rw-------`.

#### 4.5.2 Combined PEM (why required)

LiteLLM loads the client certificate into the shared `ssl.SSLContext` via `SSLContext.load_cert_chain()`, which requires the certificate **and** its unencrypted private key in one file:

```bash
cat client.crt client.key > client.pem
chmod 600 client.pem
```

> **Note on upstream versions.** Upstream LiteLLM (checked through v1.109.0) only forwards the client cert to the `httpx` transport. Its default transport (`aiohttp`) received the CA bundle but never loaded the client certificate, so mTLS silently degraded to one-way TLS and the server aborted with `certificate required`. The change in §4.5.7 fixes this centrally; with it, mTLS works on **both** transports and the aiohttp fast path is preserved.

#### 4.5.3 Complete LiteLLM YAML config template

**Where `ssl_verify` must be set.** The `openai/` provider builds its HTTP client from the global settings only (`get_ssl_configuration()` with no per-model argument), so a per-model `litellm_params.ssl_verify` is ignored on that route. Set `ssl_verify` under `litellm_settings`.

**Pick the trust file deliberately.** LiteLLM uses the `ssl_verify` file as the *only* trust anchor set; it does not add public roots to it.

- LiteLLM talks only to the GPU fleet: point `ssl_verify` at `server-ca.crt`.
- The same LiteLLM process also calls public providers (OpenAI, Anthropic, Bedrock, ...): build a combined bundle and point `ssl_verify` at it. Rebuild it whenever the image's `certifi` is upgraded, otherwise public roots go stale.

```bash
cat "$(python3 -m certifi)" /run/secrets/gpu-mtls/server-ca.crt > /run/secrets/gpu-mtls/trust-bundle.pem
```

```yaml
model_list:
  # vLLM example (neysa)
  - model_name: glm-5.3-fp8
    litellm_params:
      model: "openai/glm-5.3-fp8"
      api_base: "https://10.7.21.13:4042/v1"     # direct network IP, must be in the server cert SAN
      api_key: "os.environ/GLM_API_KEY"          # any app-layer key the backend already uses

  # SGLang example (RTX node)
  - model_name: xor-26b
    litellm_params:
      model: "openai//model"                     # actual model id in the backend
      api_base: "https://127.0.0.1:8443/v1"      # local tunnel endpoint, or direct IP
      api_key: "unused"

litellm_settings:
  ssl_verify: "/run/secrets/gpu-mtls/server-ca.crt"     # or trust-bundle.pem, see above
  ssl_certificate: "/run/secrets/gpu-mtls/client.pem"   # client cert + key

general_settings:
  master_key: "os.environ/GRID_MASTER_KEY"
```

Both knobs are read from `litellm_settings` because the proxy applies every key there with `setattr(litellm, key, value)`, and the SSL helpers read `litellm.ssl_certificate` / `litellm.ssl_verify`.

`ssl_certificate` is process-wide: LiteLLM loads it into every TLS context, so it is offered to any server that requests a client certificate. Public providers do not request one, so in practice it only reaches the GPU fleet.

#### 4.5.4 Environment variables (optional, not required)

YAML alone is sufficient when the LiteLLM build includes the §4.5.7 change. The env vars remain available as deployment-level overrides:

| Variable | Meaning | YAML equivalent |
|---|---|---|
| `SSL_CERTIFICATE` | Client cert+key PEM for mTLS | `litellm_settings.ssl_certificate` |
| `SSL_VERIFY` | Verify on/off, or CA file path | `litellm_settings.ssl_verify` |
| `SSL_CERT_FILE` | CA file used when verification is on and no CA path was given | none |

`SSL_CERTIFICATE` takes precedence over the YAML value. In production mount the files as secrets (`/run/secrets/...`) and set the paths in YAML; never bake certs into the image or commit them.

#### 4.5.5 Verification from the LiteLLM side

After `litellm --config ... --port 4000` starts, verify:

```bash
# 1) List models: should return all your YAML model entries
curl -s -H "Authorization: Bearer $GRID_MASTER_KEY" http://127.0.0.1:4000/v1/models | head

# 2) Chat completion through mTLS to the backend
curl -s -H "Authorization: Bearer $GRID_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"glm-5.3-fp8","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
     http://127.0.0.1:4000/v1/chat/completions

# 3) If public providers are also configured: one call to each, to prove the trust bundle still has public roots
```

**Failure modes to watch in LiteLLM logs:**

| Log line | Cause | Fix |
|---|---|---|
| `SSL: CERTIFICATE_VERIFY_FAILED` on a GPU route | `ssl_verify` not under `litellm_settings`, wrong path, or IP missing from server cert SAN | §4.5.3; check SAN list |
| `SSL: CERTIFICATE_VERIFY_FAILED` on a public provider | `ssl_verify` points at `server-ca.crt` only | Use `trust-bundle.pem` (§4.5.3) |
| `certificate required` / connection reset by the node | Client cert not presented (build without §4.5.7) or issued by a CA the node no longer trusts | Use a patched build; check `client.pem` was issued by the node's current `client-ca` |
| `SSLCertificateError: SSL_CERTIFICATE is configured at '...' but the file does not exist` | Path typo in YAML/env, secret not mounted. The proxy still starts and `/health/liveliness` stays green, but every request fails with a 500, public providers included, because the client cert is process-wide | Fix the path or mount |
| `SSLCertificateError: Failed to load the configured mTLS client certificate` | PEM missing the key, key encrypted, or file unreadable | Rebuild `client.pem` (§4.5.2); fix owner/permissions |
| `401 Unauthorized` | mTLS passed but `api_key` wrong | fix `api_key` in YAML |
| `Connection refused` on GPU IP | backend not listening | check backend startup args |

#### 4.5.6 Testing the two layers independently

```bash
# Direct mTLS test from the LiteLLM host (bypassing LiteLLM):
curl -v --cacert /run/secrets/gpu-mtls/server-ca.crt \
     --cert /run/secrets/gpu-mtls/client.pem \
     https://10.7.21.13:4042/v1/models

# vs broken (no client cert):
curl -v --cacert /run/secrets/gpu-mtls/server-ca.crt https://10.7.21.13:4042/v1/models
```

Proves: with client cert the handshake succeeds; without it the node refuses.

#### 4.5.7 Required code change (keep aiohttp, enable mTLS)

Only needed if the deployed LiteLLM predates this fix. The change is small and central; it does **not** disable or replace the aiohttp transport, so the existing performance profile is preserved.

**File:** `litellm/llms/custom_httpx/http_handler.py`

1. `get_ssl_client_certificate()` resolves the client cert (`SSL_CERTIFICATE` env, then `litellm.ssl_certificate`). Unset returns `None`. Set to a path that does not exist raises `SSLCertificateError`.
2. `_create_ssl_context()` takes a `client_cert` argument and calls `load_cert_chain()` on it. Any load failure (missing key, encrypted key, unreadable file, bad PEM) raises `SSLCertificateError`.
3. `get_ssl_configuration()` resolves the client cert and includes it in the cache key, so two identities never share a cached context, and a failure is raised on every call rather than cached.
4. Both `AsyncHTTPHandler.create_client` and `HTTPHandler.__init__` use the same resolver.

**Configured mTLS fails closed.** If `ssl_certificate` is set but cannot be used, requests fail with `SSLCertificateError` instead of silently falling back to one-way TLS. A strict node would reject the fallback anyway, and a permissive one must not quietly accept a connection the operator meant to authenticate.

**Why this works on aiohttp:** the resulting `ssl.SSLContext` is passed to the aiohttp `TCPConnector` and per-request `ssl=` kwarg, so the client certificate is presented during the handshake with no custom connector code and no change to concurrency/limits.

Regression tests live in `tests/test_litellm/llms/custom_httpx/test_ssl_client_certificate.py` and run a real strict-mTLS server (`CERT_REQUIRED`): valid cert accepted with the expected CN observed server-side (plain and streaming), no cert rejected, foreign-CA cert rejected, missing or corrupt PEM raises on every call, and a configured CA file stays the only trust anchor.

## 5. Maintenance

### 5.1 Planned rotation (no compromise)

- Server certs: 825-day. Reissue from `server-ca`, replace `server.{crt,key}` on the node, restart. No LiteLLM change.
- Client certs: 365-day. Issue the new cert from the *current* `client-ca` (§0.3), swap `client.pem` on the LiteLLM host, restart LiteLLM. No node change. The old cert keeps working until it expires; if that matters, use §5.2 instead.

### 5.2 Revocation of a client certificate (stolen `client.pem` or wrapper key)

Deleting the cert at LiteLLM revokes nothing: the attacker's copy still chains to a CA every node trusts. Nodes can only be told to stop trusting an issuing CA (§0.1), so revocation = rotate `client-ca` and reissue every client cert it signed. Server certs and `server-ca` are untouched. Expect a short outage per route between steps 3 and 4; for a live compromise that is the correct trade.

```bash
# 1. On the ops machine: new client CA (bump the version in the CN)
openssl req -x509 -newkey rsa:4096 -keyout client-ca-v2.key -out client-ca-v2.crt \
  -days 3650 -nodes -subj "/CN=gpu-fleet-client-ca-v2" \
  -addext "basicConstraints=critical,CA:TRUE"
chmod 600 client-ca-v2.key

# 2. Reissue every client cert from it (§0.3 with -CA client-ca-v2.crt -CAkey client-ca-v2.key):
#    LiteLLM and every Pattern C wrapper. Before overwriting, keep the stolen credential aside for step 5:
cp client.pem STOLEN.pem

# 3. Every node: trust ONLY the new client CA, then restart the engine or ingress
scp client-ca-v2.crt user@node:~/mtls/client-ca.crt

# 4. LiteLLM host and wrappers: install the reissued client.pem / client.{crt,key}, restart

# 5. Prove it, per node, from the LiteLLM host with STOLEN.pem copied over temporarily
#    (HOST, PORT, probe and server_refused as in §0.4):
r=$(probe --cert STOLEN.pem);  server_refused "$r"   || echo "FAIL: stolen cert still accepted on $HOST ($r)"
r=$(probe --cert client.pem);  [ "${r%% *}" != 000 ] || echo "FAIL: replacement cert rejected on $HOST ($r)"

# 6. On the ops machine: v2 becomes the current client CA (so later §0.3 runs sign with it), v1 is destroyed.
#    Delete STOLEN.pem from every host it was copied to.
rm -f client-ca.key STOLEN.pem
mv client-ca-v2.key client-ca.key && mv client-ca-v2.crt client-ca.crt
```

Local proof against uvicorn with the vLLM-equivalent settings (`--ssl-ca-certs <client CA> --ssl-cert-reqs 2`), same `--cacert server-ca.crt` for every call:

```
node trusts client-ca-v1:
  no client cert                   -> refused (connection reset)
  litellm-staging-v1 (later stolen) -> {"object":"list","data":[{"id":"fake"}]}
  node server cert as client       -> refused
node trusts client-ca-v2:
  stolen litellm-staging-v1        -> refused
  replacement litellm-staging-v2   -> {"object":"list","data":[{"id":"fake"}]}
```

### 5.3 Other events

- **`client-ca.key` compromise:** same as §5.2 (the attacker can mint client certs, so the CA must go).
- **`server-ca.key` compromise or a node server key leak:** new `server-ca`, reissue every server cert (§0.2), replace `server-ca.crt` (or `trust-bundle.pem`) on the LiteLLM host and wrappers. A leaked server key cannot be used as a client (§0.1, matrix case E).
- **Engine upgrades:** vLLM flags are stable, no work. SGLang: rebuild the patched image from the new tag, rerun the §0.4 matrix on one canary before fleet.
- **Adding a node:** §0.2 + appropriate pattern + matrix = ~15 min.

## 6. Known limits (deliberate scope cuts)

- mTLS secures the **LiteLLM to engine** leg and the **wrapper to engine** leg. End-user to LiteLLM remains API-key over your existing TLS termination.
- Internal RFC1918 node IPs: no public CA will issue for them, which is exactly why a private CA is used. Do not publish endpoints signed by these CAs to the internet.
- No per-certificate revocation (CRL/OCSP) on the engines; revocation granularity is the issuing CA (§5.2). Keep the number of client certs per `client-ca` small so a rotation stays cheap.
- llama.cpp / GGUF backends (`llama-server`) support server-side TLS only, with no client-cert enforcement. Put a Pattern C style ingress in front if those models must join the mTLS fleet.
