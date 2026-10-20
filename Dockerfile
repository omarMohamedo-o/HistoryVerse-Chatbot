FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache -e ".[gemini,voice,monitoring]"

FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="omarMohamedo-o" \
      org.opencontainers.image.title="HistoryVerse Microservices" \
      org.opencontainers.image.description="Model-agnostic historical AI chatbot" \
      org.opencontainers.image.authors="Omar Mohamed" \
      org.opencontainers.image.version="0.1.0"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME="/home/appuser"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgomp1 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

COPY --chown=appuser:appgroup pyproject.toml .
COPY --chown=appuser:appgroup configs ./configs
COPY --chown=appuser:appgroup src ./src
COPY --chown=appuser:appgroup dataset ./dataset

RUN mkdir -p models outputs extracted DB && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1

ENTRYPOINT ["python", "-m", "historyverse"]
