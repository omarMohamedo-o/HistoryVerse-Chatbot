"""Application settings loaded from environment variables via pydantic-settings."""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from historyverse.core.enums import DeviceType, DriftAction, LLMProvider

__all__ = ["Settings", "get_settings"]

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Central configuration for the HistoryVerse microservices ecosystem."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "historyverse-api"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    log_level: str = "INFO"

    llm_provider: LLMProvider = LLMProvider.GEMINI
    llm_model_name: str = "gemini/gemini-1.5-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8192
    llm_top_p: float = 0.95
    llm_top_k: int = 40

    llm_local_model_path: str = "./models"
    llm_model_basename: str = "llama-2-7b-chat.Q4_K_M.gguf"
    llm_auto_download: bool = True
    llm_device_type: DeviceType = DeviceType.CPU
    llm_n_gpu_layers: int = 0
    llm_n_batch: int = 512
    llm_context_window: int = 4096
    ollama_base_url: str = "http://localhost:11434"
    agent_persona_name: str = "Mora"

    google_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    huggingface_api_key: SecretStr = Field(default=SecretStr(""))

    vault_enabled: bool = False
    vault_addr: str = "http://127.0.0.1:8200"
    vault_token: SecretStr = Field(default=SecretStr(""))
    vault_role_id: str = ""
    vault_secret_id: SecretStr = Field(default=SecretStr(""))
    vault_mount: str = "historyverse"
    vault_path: str = "app"

    langfuse_public_key: SecretStr = Field(default=SecretStr(""))
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False
    langfuse_system_prompt_version: str = "v1.0"
    langfuse_user_prompt_version: str = "v1.0"
    prompt_template_name: str = "historical_chat"

    chroma_persist_dir: str = "./DB"
    source_documents_dir: str = "./dataset"
    embedding_provider: str = "google"
    embedding_model: str = "models/embedding-001"
    embedding_device: str = "cpu"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    ingest_workers: int = 4
    ingest_chunk_size_python: int = 880

    voice_enabled: bool = True
    stt_enabled: bool = True
    tts_enabled: bool = True
    vision_enabled: bool = True
    video_enabled: bool = True
    image_generation_enabled: bool = True
    image_generation_model: str = "dall-e-3"
    video_generation_enabled: bool = True
    video_generation_model: str = "luma/dream-machine"
    video_default_duration_seconds: int = 4
    video_default_fps: int = 24
    multimodal_fallback_enabled: bool = True

    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    tts_tld: str = "co.uk"

    drift_confidence_threshold: float = 0.65
    drift_max_retries: int = 2
    drift_fallback_model: str = "gemini/gemini-1.5-pro"
    ragas_min_faithfulness: float = 0.70
    ragas_min_answer_relevancy: float = 0.70
    ragas_min_context_recall: float = 0.65
    ragas_min_context_precision: float = 0.65
    drift_action: DriftAction = DriftAction.ALERT
    docker_rebuild_command: str = (
        "docker compose build historyverse-api && docker compose up -d historyverse-api"
    )

    max_conversation_history: int = 50
    redis_url: str = ""
    redis_ttl_seconds: int = 86400

    prometheus_enabled: bool = True
    alert_webhook_url: SecretStr = Field(default=SecretStr(""))
    alert_on_drift: bool = True
    alert_on_health_failure: bool = True
    alert_rate_limit_seconds: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton instance."""
    settings = Settings()
    logger.debug(
        "Application settings initialized for provider=%s model=%s",
        settings.llm_provider,
        settings.llm_model_name,
    )
    return settings
