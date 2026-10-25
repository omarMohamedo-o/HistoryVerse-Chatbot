# Peer Review & Evaluation Rubric Evidence Guide

Thank you for reviewing **Agent AI / HistoryVerse-Chatbot**. This document provides an exact 1-to-1 evidence map verifying all **10 official rubric criteria** for the **ITI × MLOps MENA Community Final Project**, adapted to our production multimodal domain: **HistoryVerse (Ancient Egyptian & Alexandrian Cultural Heritage & Museum AR Tour Guide)**.

---

## 🏛️ Project Scope & Operational Architecture

Rather than hardcoding retrieval to a single static law PDF, **HistoryVerse** applies state-of-the-art MLOps principles across a **universal, domain-agnostic agent framework**:
- **Agent 1: Historical Guide (Mora)**: Deep historical research, monument analysis, Egyptian/Alexandrian antiquity, image generation & multimodal vision.
- **Agent 2: AR Museum Tour Guide (AR Mode)**: Real-time, hands-free **Live Conversational Streaming Voice** over WebSockets (bi-directional Arabic & English, with user barge-in / interruption handling) dynamically scoped to physical museum exhibits (`artifact_label`).

All agents globally inherit:
1. **Egyptian Colloquial (Amiya) Normalizer** (`src/agent/services/nlp/colloquial.py`).
2. **Arabic Ligature, Digit & Mirrored Bracket Normalizer** (`src/agent/services/nlp/arabic.py`).
3. **Domain Boundary & Scope Refusal Gate** (`src/agent/services/rag/refusal.py`).
4. **Historical Source Grounding Guardrail** (`src/agent/services/rag/citation_guard.py`).
5. **Egyptian PII Masker** (`src/agent/services/security/pii.py`).

---

## 🚀 Quickstart: Run & Verify in 3 Commands

```bash
# 1. Install editable package
pip install -e .

# 2. Reproduce the full DVC data pipeline (extract -> validate -> chunk -> embed)
python3 scripts/dvc_pipeline.py all

# 3. Execute the full end-to-end continuous training orchestration pipeline
python3 orchestration/dags/historyverse_pipeline.py
```

---

## 📋 10-Point Rubric Evidence Map

| # | Rubric Criterion | Where to Look in This Repository | Verification Command / Quick Check |
|---|---|---|---|
| **1** | **Code & Packaging** | `pyproject.toml`, `src/agent/` | Run `pip install -e .` (clean editable installation, strict ruff linter compliance). |
| **2** | **API: Async & Pydantic** | `src/agent/api/routes/` (`/v1/chat`, `/v1/chat/structured`, `/v1/ar/ws`, `/v1/health`) | Pydantic v2 schemas in `src/agent/core/schemas.py`. Empty or invalid payload returns HTTP 422. |
| **3** | **Containerization** | `Dockerfile`, `docker-compose.yml`, `docker-compose.caddy.yml` | Multi-stage Dockerfile with non-root user and automated TLS reverse proxy via Caddy. |
| **4** | **Experiment Tracking** | `evals/chunk_experiment.py`, `evals/reports/eval_summary.md`, `evals/reports/eval_report.json` | Parameter sweep across chunk sizes and retrieval strategies with metric tables. |
| **5** | **Data Versioning (DVC)** | `dvc.yaml`, `scripts/dvc_pipeline.py`, `data/corpus/historyverse_corpus.json` | Run `python3 scripts/dvc_pipeline.py all` to reproduce extract, validate, chunk, and embed stages. |
| **6** | **CI/CD & Quality Gate** | `.github/workflows/ci.yml`, `tests/evals/test_faithfulness_perturbation.py`, `tests/regression/test_history_cases.py` | Automated PR pipeline running linting, 200+ unit tests, historical regression tests, and metamorphic perturbation checks. |
| **7** | **Production Serving** | `src/agent/api/routes/ar.py`, `src/agent/services/voice/streaming_tts.py` | Bi-directional streaming WebSocket voice service supporting user barge-in / interruption handling. |
| **8** | **Monitoring & Drift** | `configs/prometheus/alert_rules.yml`, `configs/prometheus/alerts_test.yml`, `scripts/simulate_query_drift.py` | Run `python3 scripts/simulate_query_drift.py` (100% in-domain pass, 100% out-of-domain drift caught). Run `promtool test rules configs/prometheus/alerts_test.yml`. |
| **9** | **Peer Review Guide** | This file (`PEER_REVIEW.md`), `README.ar.md` (Arabic translation) | Clear, transparent verification steps for all evaluators. |
| **10** | **Architecture & Multi-Agent** | `README.md`, `configs/prompts/` (`historical_guide.yaml`, `ar_mode.yaml`) | Complete mermaid system architecture diagram and declarative agent presets. |

---

## 🛡️ Special Domain & Safety Features

1. **Domain Boundary & Scope Refusal Gate (`src/agent/services/rag/refusal.py`)**:
   - Rejects non-historical questions (medical diagnosis, coding, football scores, cooking recipes) prior to LLM invocation to guarantee zero hallucinations.
   - Dynamically bounds the AR Tour Guide to the specific physical monument being explored (`artifact_label`).

2. **Historical Source Grounding Guardrail (`src/agent/services/rag/citation_guard.py`)**:
   - Intercepts generated answers; machine-verifies that cited historical dates, sources, and monuments actually exist in retrieved context.

3. **Arabic NLP Ligature & Digit Normalizer (`src/agent/services/nlp/arabic.py`)**:
   - Protects Lam-Alef ligatures (`\ue000`), converts Arabic-Indic numerals (`٠-٩` $\to$ `0-9`), and fixes mirrored RTL brackets.

4. **Egyptian Colloquial (Amiya) Normalizer (`src/agent/services/nlp/colloquial.py`)**:
   - Translates everyday Egyptian spoken dialect (*عايز أعرف*, *مين بنى*, *إيه حكاية*) into structured historical search concepts.

5. **Egyptian PII Masking (`src/agent/services/security/pii.py`)**:
   - Automatically sanitizes Egyptian 14-digit National IDs and mobile phone numbers before logging or embedding.
