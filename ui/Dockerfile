# syntax=docker/dockerfile:1.7

# UI container — Next.js static export served by nginx.

ARG NODE_VERSION=20.18-alpine3.20
ARG NGINX_VERSION=1.27-alpine

# ---------- builder ----------
FROM node:${NODE_VERSION} AS builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /app

# Layer the lockfile-only install above the source copy so source-only
# edits don't bust the install cache.
COPY ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

COPY ui/litellm-dashboard/ ./
RUN npm run build

# ---------- runtime ----------
FROM nginx:${NGINX_VERSION} AS runtime

# Drop the upstream default :80 server; we own the config.
RUN rm -f /etc/nginx/conf.d/default.conf \
    # nginx needs to write its pid file and response-body temp dirs even
    # when running as the unprivileged `nginx` user (uid 101); chown them
    # once at build time instead of relying on the orchestrator to mount
    # writable volumes over them.
    && mkdir -p /run /var/cache/nginx \
    && chown 101:101 /run /var/cache/nginx

# Static export → web root.
COPY --from=builder /app/out /usr/share/nginx/html

# Routing rules — see ui/nginx.conf for the full description.
COPY ui/nginx.conf /etc/nginx/nginx.conf

# Everything below runs as the unprivileged `nginx` user baked into the
# base image (uid 101, no shell), never as root.
USER 101:101

EXPOSE 3000/tcp

# nginx as PID 1 in foreground; respects SIGTERM out of the box, so
# no tini/dumb-init wrapper needed.
CMD ["nginx", "-g", "daemon off;"]
