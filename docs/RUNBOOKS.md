# Operational Runbooks

## Runbook 1: Adding or Switching an LLM Provider

1. Open `.env` (or configure your orchestrator environment variables).
2. Set `LLM_PROVIDER` to `gemini`, `openai`, `ollama`, or `local`.
3. Provide the corresponding API keys (`GOOGLE_API_KEY` or `OPENAI_API_KEY`).
4. Set `LLM_MODEL_NAME` to the desired model identifier (e.g., `gemini/gemini-1.5-pro` or `gpt-4o`).
5. For local GGUF models:
   - Place the GGUF file in `./models/` or allow automatic downloading by providing a HuggingFace repository ID in `LLM_MODEL_NAME` and filename in `LLM_MODEL_BASENAME`.
6. Restart the microservice:
   ```bash
   docker compose restart historyverse-api
   ```

## Runbook 2: Ingesting Historical Documents

1. Place new files (PDF, TXT, MD, CSV, DOCX) into `./dataset/`.
2. Trigger the ingestion endpoint:
   ```bash
   curl -X POST http://localhost:8000/v1/ingest \
     -H "Content-Type: application/json" \
     -d '{"source_directory": "./dataset"}'
   ```
3. Alternatively, invoke the CLI ingestion command:
   ```bash
   python -m historyverse.services.rag.ingest --source-dir ./dataset
   ```
4. Verify ChromaDB vector generation in the output logs.

## Runbook 3: Handling Model & RAG Drift

1. When drift is detected, the API automatically attempts up to `DRIFT_MAX_RETRIES` with temperature perturbations and context re-ranking.
2. If the confidence score remains below `DRIFT_CONFIDENCE_THRESHOLD`, the request is automatically routed to `DRIFT_FALLBACK_MODEL`.
3. If sustained drift occurs across multiple requests:
   - Check Langfuse traces for low faithfulness or high hallucination scores.
   - If `DRIFT_ACTION=docker_rebuild`, the system issues a container rebuild signal to update cached models or dependencies:
     ```bash
     docker compose build historyverse-api && docker compose up -d historyverse-api
     ```

## Runbook 4: Running Tests and Linting

1. Run Ruff linter:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```
2. Execute the test suite:
   ```bash
   uv run pytest
   ```
3. Generate coverage reports:
   ```bash
   uv run pytest --cov=historyverse --cov-report=term-missing
   ```
