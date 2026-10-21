# Code Mapping: HistoryVerse-LocalChatbot to Modern Microservices

This document maps every file and function from `HistoryVerse-LocalChatbot` (`https://github.com/omarMohamedo-o/HistoryVerse-LocalChatbot`) to its modern, model-agnostic microservices counterpart in `src/historyverse/`.

| File in `HistoryVerse-LocalChatbot` | Functionality / Line Numbers | Microservices Target in `src/historyverse/` |
| :--- | :--- | :--- |
| `constants.py` | `MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"`, `MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"` | [`src/historyverse/config/settings.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/config/settings.py) (`llm_model_basename`, `llm_local_model_path`) |
| `constants.py` | `CONTEXT_WINDOW_SIZE = 4096`, `N_BATCH = 512`, `N_GPU_LAYERS = 100` | [`src/historyverse/config/settings.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/config/settings.py) (`llm_context_window`, `llm_n_batch`, `llm_n_gpu_layers`) |
| `constants.py` | `DOCUMENT_MAP` (.txt, .md, .py, .pdf, .csv, .docx) | [`src/historyverse/services/rag/ingest.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/ingest.py) (`_get_document_map`) |
| `load_models.py` | `load_quantized_model_gguf_ggml()` using `hf_hub_download()` from HuggingFace Hub | [`src/historyverse/services/llm/providers/local.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/llm/providers/local.py) (`_ensure_gguf_available`) |
| `load_models.py` | `LlamaCpp(model_path, n_ctx, n_batch, n_gpu_layers)` execution | [`src/historyverse/services/llm/providers/local.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/llm/providers/local.py) (`generate`) |
| `prompt_template_utils.py` | LLaMA format `[INST] <<SYS>> {system_prompt} <</SYS>> {context} {question} [/INST]` | [`src/historyverse/services/llm/providers/local.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/llm/providers/local.py) (`_format_llama_prompt`) and `configs/prompts/` |
| `run_localGPT.py` | Multi-turn memory and Chroma similarity retrieval | [`src/historyverse/services/memory/manager.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/memory/manager.py) and [`src/historyverse/services/rag/retriever.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/retriever.py) |
| `ingest.py` | Multi-process ingestion with `ProcessPoolExecutor` and `ThreadPoolExecutor` | [`src/historyverse/services/rag/ingest.py`](file:///home/omar/Projects/HistoryVerse-Chatbot/src/historyverse/services/rag/ingest.py) (`DocumentIngestor.load_documents`) |
