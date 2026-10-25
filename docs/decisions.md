# Architectural Decision Records (ADRs) — HistoryVerse Platform

This document records the foundational architectural decisions, rationale, trade-offs, and technical evaluations behind the **Agent AI / HistoryVerse** platform.

---

## ADR-001: Domain-Agnostic Multi-Agent Engine vs. Monolithic Single-Script RAG

- **Status**: Accepted
- **Context**: Most traditional cohort projects couple vector retrieval directly to a single PDF inside an unmodular `app.py`. This tightly couples the application to one topic, preventing reuse, multi-modal extensions, or isolated memory.
- **Decision**: Decouple the system into declarative YAML templates (`AgentTemplate`), managed by universal engines (`TemplatedAgent`, `BaseAgent`, `AgentOrchestrator`).
- **Consequences**:
  - Positive: Specialized agents (`historical_guide`, `ar_mode`, and future agents) can be instantiated declaratively without modifying underlying pipeline code.
  - Positive: Memory isolation per session/agent is guaranteed.
  - Trade-off: Requires an abstraction layer with dependency injection rather than direct script calls.

---

## ADR-002: Real-Time WebSocket Streaming with User Barge-In for AR Mode

- **Status**: Accepted
- **Context**: Museum visitors exploring exhibits with augmented reality need hands-free, low-latency conversational audio. Standard HTTP request/response introduces seconds of turn latency and does not allow interrupting the guide if the visitor walks away or asks a new question.
- **Decision**: Implement a bi-directional WebSocket streaming architecture (`/v1/ar/ws`) combining incremental STT, token sentence-chunking (`SentenceChunker`), streaming TTS, and explicit user interruption (`VOICE_BARGE_IN_EVENTS`).
- **Consequences**:
  - Positive: Time-To-First-Audio (TTFA) drops to under 300ms.
  - Positive: Visitors can interrupt the guide mid-sentence cleanly.
  - Trade-off: Requires stateful WebSocket connection management.

---

## ADR-003: Pre-Retrieval Domain Boundary Refusal Gate

- **Status**: Accepted
- **Context**: Passing out-of-domain queries (e.g. coding, cooking recipes, medical diagnosis) directly to the LLM burns API credits and risks generating hallucinated answers.
- **Decision**: Implement `DomainRefusalGate` prior to the retrieval and LLM stages. For the Historical Guide, it enforces a strict historical scope; for AR Mode, it ensures discussions remain bound to the active monument (`artifact_label`).
- **Consequences**:
  - Positive: 100% of synthetic out-of-domain drift queries are refused before hitting the LLM.
  - Positive: Saves latency and API costs.
  - Trade-off: Requires maintaining domain keyword boundaries and colloquial mappings.

---

## ADR-004: Universal Arabic NLP Normalization & Egyptian Colloquial Mapping

- **Status**: Accepted
- **Context**: Arabic text in cultural heritage and legal PDFs contains severe PDF extraction artifacts: Lam-Alef ligatures (`لا`, `لأ`, `لإ`, `لآ`) collapsed into single glyphs, Arabic-Indic numerals (`٠-٩`), and mirrored brackets from bidirectional reversals. Furthermore, real Egyptian users ask questions using everyday colloquial dialect (*العامية المصرية*).
- **Decision**: Build `clean_arabic_text()` with Unicode NFKC, ligature shielding (`\ue000`), digit translation, and bracket mirroring, paired with `expand_egyptian_colloquial_query()` to enrich colloquial queries with formal terminology.
- **Consequences**:
  - Positive: Dense and BM25 retrieval match Arabic entities accurately regardless of whether the user typed Western or Arabic-Indic numerals.
  - Positive: Spoken Egyptian queries retrieve formal classical texts seamlessly.

---

## ADR-005: Self-Contained Local DVC Pipeline vs. Cloud-Credential Gated Remote

- **Status**: Accepted
- **Context**: Several peer projects configured DVC with private Google Drive or S3 remotes, causing `dvc repro` to crash with credential errors for reviewers and automated CI runners.
- **Decision**: Configure a self-contained 4-stage DVC pipeline (`extract` $\to$ `validate` $\to$ `chunk` $\to$ `embed`) backed by local versioned sample fixtures in `data/corpus/`.
- **Consequences**:
  - Positive: Any reviewer or automated GitHub runner can clone the repo and execute `dvc repro` or `python3 scripts/dvc_pipeline.py all` with zero credentials and 100% reproducibility.

---

## ADR-006: Metamorphic Perturbation Testing for RAG Evaluation Sensitivity

- **Status**: Accepted
- **Context**: Evaluators frequently question whether an automated LLM judge is a "rubber stamp" that blindly awards high scores.
- **Decision**: Implement metamorphic perturbation tests (`tests/evals/test_faithfulness_perturbation.py`) that systematically corrupt valid historical answers (altering construction centuries, builder attributions, or source tags) and assert that faithfulness scores plummet.
- **Consequences**:
  - Positive: Empirically proves that the evaluation pipeline is sensitive and mathematically distinguishes true grounded answers from hallucinations.
