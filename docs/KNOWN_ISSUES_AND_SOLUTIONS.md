# Known Issues, Architectural Bugs, and Permanent Solutions

This document records the exact bugs, dependency incompatibilities, and architectural challenges identified during the refactoring and unification of HistoryVerse, along with their permanent solutions and verification tests.

---

### Issue 1: `TypeError: unhashable type: 'Settings'` in FastAPI Dependencies
- **Symptoms**: Integration tests and API requests to `/v1/chat`, `/v1/evaluate`, `/v1/voice` failed with HTTP 500 (`Internal Server Error`).
- **Root Cause**: FastAPI dependency functions (e.g. `get_llm_provider`, `get_drift_detector`) were decorated with `@lru_cache(maxsize=1)` while accepting `settings: Settings = Depends(get_app_settings)`. Because Pydantic v2 `BaseSettings` is a mutable class, `lru_cache` raised a `TypeError` when hashing function arguments.
- **Permanent Solution**: Removed `@lru_cache` from functions with complex arguments in [`src/historyverse/api/dependencies.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/api/dependencies.py). Implemented an explicit `_SINGLETON_CACHE: dict[str, Any]` with thread-safe singleton initialization and a `reset_dependencies_cache()` utility for test isolation.
- **Verification**: Verified by [`tests/integration/test_api.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/integration/test_api.py) passing 100%.

---

### Issue 2: LangChain 0.3+ Package Reorganization (`Document` and `RecursiveCharacterTextSplitter`)
- **Symptoms**: `ModuleNotFoundError: No module named 'langchain.docstore'` and `ModuleNotFoundError: No module named 'langchain.text_splitter'`.
- **Root Cause**: Upstream LangChain 0.3+ split monolith modules into modular packages (`langchain_core` and `langchain_text_splitters`). Legacy import paths `langchain.docstore.document` and `langchain.text_splitter` were deprecated and removed.
- **Permanent Solution**:
  - Replaced `from langchain.docstore.document import Document` with `from langchain_core.documents import Document` across [`ingest.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/ingest.py) and [`retriever.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/retriever.py).
  - Replaced `from langchain.text_splitter import Language, RecursiveCharacterTextSplitter` with `from langchain_text_splitters import Language, RecursiveCharacterTextSplitter`.
- **Verification**: Tested by [`tests/unit/test_ingest_strategies.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_ingest_strategies.py) and ingestion pipeline tests.

---

### Issue 3: Incomplete BaseLLMProvider Import in Services Init
- **Symptoms**: `ModuleNotFoundError: No module named 'historyverse.services.llm.providers.base'` when importing from `historyverse.services.llm`.
- **Root Cause**: `src/historyverse/services/llm/__init__.py` attempted to import `BaseLLMProvider` from `historyverse.services.llm.providers.base` instead of the canonical domain location `historyverse.core.base`.
- **Permanent Solution**: Corrected import in [`src/historyverse/services/llm/__init__.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/llm/__init__.py) to `from historyverse.core.base import BaseLLMProvider`.
- **Verification**: Verified by all LLM factory tests in [`tests/unit/test_llm_factory.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_llm_factory.py).

---

### Issue 4: Hallucination Detector Token Matching with Punctuation
- **Symptoms**: `test_grounded_answer_low_hallucination_score` failed with `score == 0.40` when ground-truth answers contained punctuation (e.g. `Serapis.` vs context `Serapis,`).
- **Root Cause**: Raw string splitting (`.split()`) retained trailing punctuation marks, causing exact token matching between response words and context words to miss valid grounded historical terms.
- **Permanent Solution**: Applied punctuation stripping (`.strip(".,!?:;'\"()")`) to both response words and context vocabulary in [`src/historyverse/services/evaluation/hallucination.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/evaluation/hallucination.py), increasing grounding accuracy.
- **Verification**: Verified by [`tests/unit/test_hallucination.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_hallucination.py).

---

### Issue 5: Ollama Detection Collision with Default GGUF Basename
- **Symptoms**: `test_local_provider_unified_routing` failed when routing `LLMProvider.LOCAL` with `llm_model_name="llama3.2"`.
- **Root Cause**: `Settings` defaults `llm_model_basename` to `"llama-2-7b-chat.Q4_K_M.gguf"`. When a user specified only `llm_model_name="llama3.2"`, checking `_model_basename.endswith(".gguf")` evaluated to True and wrongly bypassed Ollama mode.
- **Permanent Solution**: Updated `_detect_ollama_mode()` in [`src/historyverse/services/llm/providers/local.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/llm/providers/local.py) to check `llm_model_name` directly: single-tag names without `/` and not ending in `.gguf` automatically route to the Ollama engine.
- **Verification**: Verified by [`tests/unit/test_llm_factory.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_llm_factory.py).

---

### Issue 6: Redundant Ingress Sidecars (Certbot vs. Caddy)
- **Problem**: Having both Nginx + Certbot sidecars (`docker-compose.ssl.yml`) alongside Caddy (`docker-compose.caddy.yml`) caused configuration duplication and unnecessary container overhead.
- **Solution**: Removed `docker-compose.ssl.yml`. Standardized on **Caddy** for Docker Compose deployments, providing built-in automatic Let's Encrypt / ZeroSSL TLS, HTTP/3, and reverse proxying with a single container and zero Certbot cron jobs.

---

### Issue 7: Direct LLM Injection in Structured Output Endpoint
- **Symptoms**: `AttributeError: 'RAGPipeline' object has no attribute '_drift_detector'` on `/v1/chat/structured`.
- **Root Cause**: The `/v1/chat/structured` route attempted to access `pipeline._drift_detector._primary` directly rather than leveraging standard FastAPI dependency injection.
- **Permanent Solution**: Updated [`src/historyverse/api/routes/chat.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/api/routes/chat.py) to declare `provider: BaseLLMProvider = Depends(get_llm_provider)` directly in the endpoint signature, maintaining loose coupling and mockability.
- **Verification**: Verified by [`tests/integration/test_unlearning_api.py::test_chat_structured_api_route`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/integration/test_unlearning_api.py).

---

### Issue 8: Non-Strict `zip()` during Vector Store Document Deletion
- **Symptoms**: Ruff B905 error flagging `zip()` call without explicit `strict=` argument.
- **Root Cause**: ChromaDB collection queries return variable dictionary lengths; invoking `zip()` without `strict=` parameter triggered Python 3.10+ lint safety warnings.
- **Permanent Solution**: Passed `strict=False` explicitly in [`src/historyverse/services/rag/retriever.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/retriever.py).
- **Verification**: Verified by clean `ruff check src/ tests/`.

---

### Issue 9: Mutable Default Class Attribute in `UnlearningValidator`
- **Symptoms**: Ruff RUF012 warning regarding mutable list default on class attribute.
- **Root Cause**: `RETRACTION_INDICATORS` was declared as a mutable `list[str]` at class definition scope.
- **Permanent Solution**: Converted `RETRACTION_INDICATORS` to an immutable `tuple[str, ...]` in [`src/historyverse/services/unlearning/validator.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/unlearning/validator.py).
- **Verification**: Verified by clean `ruff check src/ tests/`.

---

### Issue 10: Multi-Agent Context Bleed & Memory Cross-Contamination
- **Problem**: When multiple agents (e.g. Mora, Archival Researcher, Critic) interacted with the same user session, their conversation turns and scratchpads shared the same session key, contaminating agent roles.
- **Solution**: Implemented `AgentMemory` in [`src/historyverse/services/memory/agent_memory.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/memory/agent_memory.py) which automatically namespaces session identifiers as `{agent_id}::{session_id}`. Created `UnlearningEngine` and `UnlearningValidator` to enable selective forgetting of retracted facts with automated adversarial probe testing.
- **Verification**: Verified by [`tests/unit/test_agent_memory.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_agent_memory.py) and [`tests/unit/test_unlearning.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/tests/unit/test_unlearning.py).
