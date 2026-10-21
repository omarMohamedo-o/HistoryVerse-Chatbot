# Caddy, Redis, and Smartheal Operations Guide

This guide documents the edge reverse proxy (Caddy), distributed session memory & caching (Redis), and automated self-healing orchestration (Smartheal & Heartbeat) for HistoryVerse.

## 1. Caddy Reverse Proxy & Automatic TLS

Caddy provides an edge gateway for Docker Compose deployments with automated SSL certificate provisioning via Let's Encrypt / ZeroSSL.

### Features
- Built-in ACME client (no Certbot sidecar or crontabs required).
- Native HTTP/2 and HTTP/3 (QUIC) support.
- Compression (`zstd`, `gzip`) and hardened security headers (`HSTS`, `X-Content-Type-Options`, `X-Frame-Options`).
- Clean reverse proxy handling for `/api/*`, `/v1/*`, `/healthz`, `/metrics`, and OpenAPI docs.

### Running with Docker Compose
```bash
# Start API stack
docker compose up -d

# Start Caddy TLS edge
docker compose -f docker-compose.caddy.yml up -d
```

---

## 2. Redis Distributed Session Memory & Fallback

`ConversationMemory` supports Redis persistence for multi-worker and multi-container deployments.

### Environment Configuration
- `REDIS_URL`: Connection string (e.g., `redis://redis:6379/0`). Defaults to empty (`""`).
- `REDIS_TTL_SECONDS`: Session expiration time (default: `86400` / 24 hours).

### Resilient Fallback Architecture
- If `REDIS_URL` is set and Redis is reachable, all conversation turns are saved under `historyverse:session:{conversation_id}`.
- If Redis is unavailable, unconfigured, or suffers a transient outage, `ConversationMemory` automatically falls back to thread-safe local in-memory storage without dropping user queries or throwing errors.

---

## 3. Smartheal Automated Health Orchestration

Adapted from `pipeline-infra/services/smartheal`, the `smartheal` daemon monitors Docker container health status and handles controlled restarts.

### Key Behaviors
- **Dependency Awareness**: Respects service startup order (`historyverse-api` depends on `redis`).
- **Exponential Backoff**: Prevents restart flapping.
- **Quarantine Window**: If a service exceeds 5 restarts within 10 minutes, it enters a 30-minute quarantine to protect system stability.

---

## 4. Heartbeat External Dead-Man's Switch

Located at `services/heartbeat/`:
- Runs a 60-second polling loop verifying that Prometheus `/ready` is passing and no critical alerts are firing.
- Dispatches a ping to an external uptime check (e.g. Healthchecks.io, BetterUptime).
- Withholds the ping if internal health checks fail, ensuring external alerting when the host or services degrade.
