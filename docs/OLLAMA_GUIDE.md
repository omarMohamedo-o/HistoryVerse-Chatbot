# Ollama Integration Guide for HistoryVerse

HistoryVerse provides first-class support for Ollama, enabling 100% local, offline, and private AI execution for text chat, RAG embeddings, and multimodal vision.

## 1. Supported Models

| Task | Recommended Ollama Models |
| :--- | :--- |
| **Historical Chat & Reasoning** | `llama3.2`, `llama3.1:8b`, `mistral`, `gemma2:9b`, `qwen2.5:7b`, `phi3:mini` |
| **Multimodal Vision (Artifact/Site Analysis)** | `llava:13b`, `llava:7b`, `llama3.2-vision`, `bakllava` |
| **RAG Vector Embeddings** | `nomic-embed-text`, `all-minilm`, `bge-m3` |

---

## 2. Configuration via `.env`

To run HistoryVerse with Ollama, set the following environment variables:

```bash
# Model selection
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama3.2

# Ollama server connection (default: http://localhost:11434, or http://ollama:11434 in Docker)
OLLAMA_BASE_URL=http://localhost:11434

# Auto-pull models if missing from Ollama
LLM_AUTO_DOWNLOAD=true

# Optional: Fully local RAG embeddings with Ollama
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

---

## 3. Running with Docker Compose

An `ollama` service is included in `docker-compose.yml`:

```bash
# Start all microservices including Ollama and Redis
docker compose up -d

# Check Ollama logs and pulled models
docker compose exec ollama ollama list

# Pull a model manually (if auto-download is disabled)
docker compose exec ollama ollama pull llama3.2
```

---

## 4. Key Architectural Features

- **Automated Pulling via REST API**: When `LLM_AUTO_DOWNLOAD=true`, `OllamaProvider` automatically queries `GET /api/tags` and dispatches `POST /api/pull` if the model is missing.
- **Multimodal Support**: When analyzing images or video keyframes (`/v1/vision/analyze`), `OllamaProvider` encodes the frames as base64 and invokes Ollama's vision endpoints directly.
- **Dual-Engine Execution**: Uses LiteLLM for high-throughput streaming and completions, with automatic fallback to direct REST API calls.
