# syntax=docker/dockerfile:1.7

# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:103eb3f4444c68ea2453bf3aad09d860eaa5a698effb3e656cd607f630f0e46d

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:103eb3f4444c68ea2453bf3aad09d860eaa5a698effb3e656cd607f630f0e46d
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a
# Pinned by digest like the other base images; bump explicitly on Node upgrades.
ARG UI_BUILD_IMAGE=node:20.18-alpine3.20@sha256:3488b10bf958af7125a176419d2d8a9937d895bf124012aae811651988d2ffe6

FROM $UV_IMAGE AS uvbin

# Admin UI builder. Pinned to the build platform so the architecture-independent
# Next.js static export compiles once natively even in a multi-arch build,
# instead of once per target arch under QEMU.
FROM --platform=$BUILDPLATFORM $UI_BUILD_IMAGE AS ui-builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /ui

COPY ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --prefer-offline

COPY ui/litellm-dashboard/ ./
RUN npm run build

# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

WORKDIR /app
USER root

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx

# ca-certificates: eMudra remediation — drop emSign ECC Root CA - G3 / AAA
# Certificate Services from the system bundle (>=20260909-r3).
# nodejs>=26.12.1: patched Node TLS trust (Prisma CLI makes outbound TLS).
RUN apk add --no-cache \
    bash \
    gcc \
    python3 \
    python3-dev \
    rust \
    openssl \
    openssl-dev \
    'ca-certificates>=20260909-r3' \
    'nodejs>=26.12.1' \
    npm \
    libsndfile

# eMudra / emSign G3 advisory: the ca-certificates version number is not a
# trustworthy signal on Wolfi (20260909-r3 shipped a stale bundle still
# anchoring emSign ECC Root CA - G3). Strip distrusted roots by content and
# fail the build if any survive. Must run BEFORE the TLS downloads below
# (uv sync, prisma, npm). Self-verifying — see docker/strip_distrusted_cas.py.
COPY docker/strip_distrusted_cas.py /usr/local/bin/strip_distrusted_cas.py
RUN python3 /usr/local/bin/strip_distrusted_cas.py

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    PRISMA_BINARY_CACHE_DIR=/app/.cache/prisma-python/binaries \
    XDG_CACHE_HOME=/app/.cache

# Copy dependency metadata first for layer caching
COPY pyproject.toml uv.lock ./
COPY enterprise/pyproject.toml enterprise/
COPY litellm-proxy-extras/pyproject.toml litellm-proxy-extras/

# Install third-party dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --frozen --no-install-project --no-install-workspace --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3.13

# Copy full source tree
COPY . .

# Replace the committed UI bundle with the one built from this exact source.
# Clearing first drops the committed bundle's content-hashed chunks that COPY
# would otherwise leave behind alongside the fresh ones.
RUN rm -rf litellm/proxy/_experimental/out
COPY --from=ui-builder /ui/out/. litellm/proxy/_experimental/out/

# Build Admin UI before final sync (applies the enterprise color override when present)
RUN sed -i 's/\r$//' docker/build_admin_ui.sh && chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Install project and workspace packages (fast - deps already cached)
RUN uv sync --frozen --no-default-groups --no-editable \
    --extra proxy \
    --extra proxy-runtime \
    --extra extra_proxy \
    --extra semantic-router \
    --python python3.13

RUN prisma generate --schema=./schema.prisma

# eMudra / emSign G3: second pass after uv sync + prisma, so vendored bundles
# (botocore/cacert.pem ships AAA Certificate Services) in the venv are covered.
COPY docker/strip_distrusted_cas.py /usr/local/bin/strip_distrusted_cas.py
RUN python3 /usr/local/bin/strip_distrusted_cas.py

RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh && \
    sed -i 's/\r$//' docker/prod_entrypoint.sh && chmod +x docker/prod_entrypoint.sh

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

USER root

# node (without npm) is required by the prisma CLI at runtime.
# ca-certificates / nodejs floors: eMudra CA remediation (emSign G3 / AAA roots).
RUN apk add --no-cache \
    bash \
    openssl \
    tzdata \
    'ca-certificates>=20260909-r3' \
    'nodejs>=26.12.1' \
    python3 \
    libsndfile

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PRISMA_BINARY_CACHE_DIR=/app/.cache/prisma-python/binaries \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    PRISMA_SKIP_POSTINSTALL_GENERATE=1 \
    PRISMA_HIDE_UPDATE_MESSAGE=1 \
    PRISMA_ENGINES_CHECKSUM_IGNORE_MISSING=1 \
    PRISMA_OFFLINE_MODE=true

# Copy only what runtime needs. The application is installed inside the venv;
# the rest of the builder's /app is source and build metadata that must not
# ship (manifest-scanning tools attribute everything in it to this image).
# entrypoint.sh invokes litellm/proxy/prisma_migration.py by source path.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/docker /app/docker
COPY --from=builder /app/schema.prisma /app/schema.prisma
COPY --from=builder /app/litellm/proxy/prisma_migration.py /app/litellm/proxy/prisma_migration.py
# enterprise/ is imported by source path at runtime (proxy_cli puts the
# working directory on sys.path; litellm/proxy/hooks resolves
# enterprise.enterprise_hooks from it)
COPY --from=builder /app/enterprise /app/enterprise
# Prisma caches live under /app/.cache (XDG_CACHE_HOME /
# PRISMA_BINARY_CACHE_DIR, set in the builder too) so the runtime user,
# which has no writable $HOME, can still read them. Copy only the Prisma
# subdir — copying the whole /app/.cache drags in the uv build cache
# (~660 MB, includes a setuptools wheel that surfaces as a CVE finding
# even though it's not on the runtime sys.path).
COPY --from=builder /app/.cache/prisma-python /app/.cache/prisma-python

# eMudra / emSign G3: re-run now that the venv is present, so vendored bundles
# (botocore/cacert.pem still ships AAA Certificate Services) get stripped too.
# Self-verifying — fails the build if any distrusted root survives anywhere.
COPY docker/strip_distrusted_cas.py /usr/local/bin/strip_distrusted_cas.py
RUN python3 /usr/local/bin/strip_distrusted_cas.py

RUN find /app/.venv -type f -path "*/tornado/test/*" -delete && \
    find /app/.venv -type d -path "*/tornado/test" -delete

# Runtime must never be root (echo-scan Issues #15/#16). nobody (uid 65534)
# has no writable $HOME, so give it the paths it must write: the prisma client
# dir (runtime `prisma generate` in entrypoint flows) and $HOME (/app) for
# libraries that persist state under ~/.config (e.g. xAI OAuth tokens).
# LITELLM_NON_ROOT is intentionally unset: this image serves the packaged UI
# from the venv, and the flag would point it at /var/lib/litellm/ui, which
# this image does not stage.
RUN mkdir -p /nonexistent && \
    chmod -R a+rX /app && \
    PRISMA_PATH=$(python -c "import os, prisma; print(os.path.dirname(prisma.__file__))") && \
    chmod -R a+rwX "$PRISMA_PATH" && \
    chmod -R a+rwX /app/.cache
USER 65534

EXPOSE 4000/tcp

ENTRYPOINT ["docker/prod_entrypoint.sh"]
CMD ["--port", "4000"]
