# Code Map

This document serves as the architectural index of the HistoryVerse codebase.

## Directory Overview

```
.
├── configs/                       # Declarative configurations and versioned templates
│   ├── evaluation/                # Ragas and benchmark evaluation configurations
│   ├── loki/                      # Loki log collection configuration
│   ├── prometheus/                # Prometheus scraper & alert rules
│   ├── promtail/                  # Promtail container log forwarder config
│   └── prompts/                   # Fallback prompt templates (YAML) with versioning
├── dataset/                       # Primary historical documents (PDFs, TXT, MD, etc.)
├── docs/                          # Operational, architectural, and governance documentation
│   ├── bugs/                      # Bug tracking fragments
│   ├── changelog/                 # Release notes and changelog fragments
│   ├── BEST_PRACTICES.md          # 25 production best practices documentation
│   ├── CODE_MAP.md                # This file
│   ├── DISK_GUARDIAN.md           # Disk guard and storage maintenance documentation
│   ├── DOCKER_BEST_PRACTICES.md   # Container security and hardening documentation
│   ├── GITHUB_OPERATING_MODEL.md  # Git workflow and policy guidelines
│   ├── MONITORING_AND_SECURITY.md # Prometheus, Loki, Trivy, Snyk documentation
│   └── RUNBOOKS.md                # System runbooks for ops and maintenance
├── extracted/                     # Intermediate or cached text chunks
├── models/                        # Local model weights cache (GGUF, HuggingFace)
├── outputs/                       # Transient generated outputs, TTS audio, evaluations
├── scripts/                       # Operational scripts and utilities
│   ├── docker-clean.sh            # Safe Docker resource maintenance and prune script
│   └── docs_log.py                # Changelog and bug fragment generator
├── services/                      # Operational infrastructure companion services
│   └── disk-guardian/             # Automated host disk and container storage guard
├── src/historyverse/              # Core application package
│   ├── api/                       # FastAPI router, middleware, and dependency injection
│   │   ├── routes/                # Endpoint definitions (/chat, /ingest, /voice, etc.)
│   │   ├── app.py                 # App factory and lifespan handlers
│   │   ├── dependencies.py        # Dependency injection providers
│   │   └── middleware.py          # Request logging and exception handling middleware
│   ├── config/                    # Pydantic settings and environment management
│   │   └── settings.py            # Strongly-typed environment configuration
│   ├── core/                      # Core interfaces, domain types, and base classes
│   │   ├── base.py                # Abstract Base Classes (LLM, Embedding, STT, TTS)
│   │   ├── enums.py               # Enums (LLMProvider, ConversationType, DriftAction)
│   │   ├── exceptions.py          # Exception hierarchy
│   │   └── schemas.py             # Pydantic request and response schemas
│   └── services/                  # Business logic services
│       ├── alerting/              # Incident notification and webhook dispatcher
│       ├── evaluation/            # Ragas metric evaluation and hallucination scoring
│       ├── langfuse/              # Langfuse prompt registry and telemetry tracing
│       ├── llm/                   # Provider implementations, factory, and drift detector
│       │   └── providers/         # Concrete providers (Gemini, OpenAI, Local)
│       ├── memory/                # Thread-safe conversation history manager
│       ├── monitoring/            # Prometheus metric instruments
│       ├── rag/                   # Ingestion, ChromaDB vector store, and QA pipeline
│       ├── vision/                # Multimodal image and video analysis
│       └── voice/                 # Speech-to-Text and Text-to-Speech handlers
└── tests/                         # Pytest test suite
    ├── conftest.py                # Shared fixtures and mock providers
    ├── integration/               # API and pipeline integration tests
    └── unit/                      # Isolated unit tests
```

## Key Module Responsibilities

| Module | Purpose |
|---|---|
| `historyverse.config.settings` | Loads, validates, and manages all environment variables via Pydantic Settings. |
| `historyverse.core.base` | Defines abstract base classes ensuring modular, pluggable providers. |
| `historyverse.core.schemas` | Pydantic v2 schemas for all API payloads and internal data transfer. |
| `historyverse.services.llm.factory` | Instantiates providers based on config; ensures local models are downloaded. |
| `historyverse.services.llm.drift` | Monitors model accuracy; triggers automatic re-run or docker rebuild on drift. |
| `historyverse.services.alerting` | Dispatches rate-limited operational alerts to webhooks and incident channels. |
| `services/disk-guardian` | Continuous daemon monitoring storage and executing autonomous container pruning. |
| `scripts/docker-clean.sh` | Operational script for manual or cron Docker pruning. |
| `historyverse.api.app` | Configures FastAPI app, mounts routes, and applies middleware. |
