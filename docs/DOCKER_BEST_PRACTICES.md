# Docker Production Best Practices

This document outlines the container engineering and security best practices implemented in HistoryVerse.

| # | Best Practice | Implementation in HistoryVerse |
|---|---|---|
| 1 | **Multi-Stage Builds** | Build dependencies and compilation occur in `builder` stage; lean virtual environment copied to `runtime`. |
| 2 | **Minimal & Pinned Base Image** | Pinned to `python:3.11-slim-bookworm` to minimize attack surface and image size while ensuring glibc compatibility. |
| 3 | **Non-Root User Execution** | Application runs under dedicated unprivileged user `appuser` (`UID 10001`, `GID 10001`). |
| 4 | **Optimal Layer Caching** | Rarely-changing system packages and `pyproject.toml` dependencies cached before copying application source code. |
| 5 | **Native Container Healthcheck** | Standardized `HEALTHCHECK` probing `curl -f http://localhost:8000/health` with retry and start-period grace windows. |
| 6 | **Proper Signal Handling** | `ENTRYPOINT ["python", "-m", "historyverse"]` (exec form) ensures PID 1 properly receives and handles `SIGTERM` / `SIGINT`. |
| 7 | **Strict `.dockerignore`** | Prevents `.git`, `.venv`, `.env`, local databases, caches, and test artifacts from leaking into build context. |
| 8 | **Zero Secrets in Images** | No API keys baked into images; injected dynamically via `.env` or container orchestrators at runtime. |
| 9 | **Resource Limits & Quotas** | `docker-compose.yml` configures hard limits (`2.0 CPUs`, `4GB RAM`) and reservations (`0.5 CPUs`, `512MB RAM`). |
| 10 | **Read-Only Dataset Mounts** | Static raw reference documents mounted with `:ro` (`./dataset:/app/dataset:ro`) to protect source data. |
| 11 | **Isolated State Volumes** | Mutable state directories (`DB/`, `models/`, `outputs/`) isolated through bind and volume mounts. |
| 12 | **Package Cache Cleanups** | Single-run layer cleanup via `apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*`. |
| 13 | **No Recommends Installation** | `apt-get install -y --no-install-recommends` prevents pulling unnecessary companion packages. |
| 14 | **Log Rotation Configuration** | JSON-file logging driver with `max-size: 10m` and `max-file: 3` to prevent disk saturation. |
| 15 | **Standard OCI Metadata Labels** | Maintainer, version, description, and source repository documented in image labels. |
