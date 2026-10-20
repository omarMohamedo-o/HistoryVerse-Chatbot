# INTENT - What We Are Building, How, and the Rules

## 1. Intent & Mission
HistoryVerse is a high-reliability, model-agnostic historical AI chatbot microservice suite.
It unifies local document knowledge (Alexandria, Egypt, Greco-Roman Museum, general world history)
with state-of-the-art LLMs (Google Gemini, OpenAI, or fully local GGUF/Ollama LLaMA models) via:
- Multi-format ingestion (PDF, TXT, MD, DOCX, CSV, Excel, Images, Code) into ChromaDB
- Model-agnostic LLM interface with automatic fallback and download capabilities
- Prompt and configuration versioning through Langfuse and declarative YAML
- Automated drift detection, accuracy verification, and re-run triggers
- RAG evaluation using Ragas metrics (faithfulness, answer relevancy, context precision, context recall)
- Speech-to-Text (Faster-Whisper) and Text-to-Speech (gTTS) pipeline
- Multimodal Vision capabilities for artifact and monument analysis
- Production-grade FastAPI endpoints with structured logging and Pydantic validation

## 2. Core Architectural Invariants
1. **Model Agnostic & Local Resilience**:
   Any LLM supported by LiteLLM or local inference engines (Ollama, llama-cpp) can be selected via configuration. If a local model is specified but not present on the filesystem, it is fetched automatically.
2. **Zero Hardcoded Values**:
   All parameters (temperatures, chunk sizes, token limits, model names, endpoints) must originate from `Settings` loaded via `pydantic-settings` or versioned Langfuse prompt registries.
3. **Structured Logging Exclusively**:
   Standard Python `logging` or `structlog` must be used across all modules. Direct `print()` statements are forbidden.
4. **Drift Detection & Quality Safeguard**:
   Every generated response undergoes confidence scoring and hallucination assessment. When responses fall below acceptable thresholds, an automatic re-run with adjusted parameters or fallback models is triggered, and persistent drift triggers operator alerts or container recreation hooks.
5. **Experiment Tracking**:
   Prompts, latencies, token consumption, context citations, and Ragas scores are tracked in Langfuse.
6. **Clean OOP & Type Safety**:
   Interfaces are defined with Abstract Base Classes (ABCs), schemas validated through Pydantic v2, and all functions typed with strict Python 3.11+ type hints.
7. **Single Operator Model**:
   All commits and contributions are credited to `@omarMohamedo-o`. No AI assistant attribution or metadata is permitted anywhere in git logs, code, or documentation.
