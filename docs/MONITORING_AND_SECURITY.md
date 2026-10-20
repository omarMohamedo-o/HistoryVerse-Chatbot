# Monitoring, Observability, and Security Architecture

This document describes the operational observability and automated security controls in HistoryVerse.

## 1. Observability Stack

- **Prometheus**: Scrapes `/metrics` from `historyverse-api` every 15s. Records request latencies, LLM call timings, token consumption, and drift occurrences.
- **Grafana**: Visualizes metrics dashboards for operational latencies, confidence trends, and system loads.
- **Loki & Promtail**: Centralized log streaming and indexing from Docker container logs without manual log parsing.
- **Langfuse**: Tracing for LLM prompts, latencies, tokens, Ragas metrics, and hallucination scores.

### Starting the Monitoring Stack

```bash
docker network create historyverse-network || true
docker compose up -d
docker compose -f docker-compose.monitoring.yml up -d
```

- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000` (default credentials: `admin` / `admin`)
- Loki API: `http://localhost:3100`

## 2. Health & Readiness Probes

| Endpoint | Type | Purpose |
|---|---|---|
| `GET /health` | Liveness | Basic HTTP responsiveness check. |
| `GET /health/live` | Liveness | Container orchestrator liveness probe. |
| `GET /health/ready` | Readiness | Deep probe verifying LLM availability, Chroma accessibility, and disk space. |
| `GET /metrics` | Observability | Prometheus metric exporter. |

## 3. Automated Alerting System

Configured via `AlertNotifier` (`src/historyverse/services/alerting/`):
- Triggers on sustained low-confidence model drift (`HighModelDriftRate`).
- Triggers on readiness check failures or service degradation (`ServiceDegraded`).
- Dispatches webhook payloads with rate-limiting controls to prevent notification storms.

## 4. Security Gates (Trivy & Snyk)

Configured via `.github/workflows/security.yml`:
- **Trivy**: Scans the filesystem, dependencies, and container image for CVEs and misconfigurations. Blocks CRITICAL and HIGH severity vulnerabilities.
- **Snyk**: Evaluates open-source packages and performs Static Application Security Testing (SAST) for Python code flaws.
