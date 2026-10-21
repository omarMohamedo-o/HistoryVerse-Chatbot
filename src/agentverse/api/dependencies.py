"""FastAPI dependency injection container."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends

from agentverse.config.settings import Settings, get_settings
from agentverse.core.base import BaseLLMProvider
from agentverse.services.evaluation.hallucination import HallucinationDetector
from agentverse.services.evaluation.ragas_eval import RagasEvaluator
from agentverse.services.langfuse.prompts import PromptManager
from agentverse.services.llm.drift import DriftDetector
from agentverse.services.llm.factory import LLMFactory
from agentverse.services.memory.manager import ConversationMemory
from agentverse.services.rag.file_chat import FileChatService
from agentverse.services.rag.ingest import DocumentIngestor
from agentverse.services.rag.pipeline import RAGPipeline
from agentverse.services.rag.retriever import ChromaRetriever
from agentverse.services.vision.analyzer import ImageAnalyzer
from agentverse.services.vision.generator import ImageGenerator
from agentverse.services.voice.stt import WhisperSTT
from agentverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)

__all__ = [
    "get_agent_memory_manager",
    "get_app_settings",
    "get_ar_agent",
    "get_conversation_memory",
    "get_document_ingestor",
    "get_drift_detector",
    "get_file_chat_service",
    "get_hallucination_detector",
    "get_historyverse_ar_mode_agent",
    "get_image_analyzer",
    "get_image_generator",
    "get_llm_provider",
    "get_prompt_manager",
    "get_rag_pipeline",
    "get_ragas_evaluator",
    "get_retriever",
    "get_stt_provider",
    "get_tts_provider",
    "get_unlearning_engine",
    "get_video_generator",
]

# Process-level singleton cache for heavy microservice components
_SINGLETON_CACHE: dict[str, Any] = {}


def reset_dependencies_cache() -> None:
    """Clear all cached singletons for testing isolation."""
    _SINGLETON_CACHE.clear()


def get_app_settings() -> Settings:
    """Return application settings."""
    return get_settings()


def get_llm_provider(settings: Settings = Depends(get_app_settings)) -> BaseLLMProvider:
    """Return the configured primary LLM provider."""
    if "llm_provider" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["llm_provider"] = LLMFactory.create(settings)
    return _SINGLETON_CACHE["llm_provider"]


def get_drift_detector(
    settings: Settings = Depends(get_app_settings),
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> DriftDetector:
    """Return the drift detector configured with primary and fallback providers."""
    if "drift_detector" not in _SINGLETON_CACHE:
        fallback_provider = None
        if settings.drift_fallback_model:
            try:
                fallback_provider = LLMFactory.create(settings)
            except Exception as exc:
                logger.warning("Could not instantiate fallback provider: %s", exc)

        _SINGLETON_CACHE["drift_detector"] = DriftDetector(
            primary_provider=provider,
            fallback_provider=fallback_provider,
            settings=settings,
        )
    return _SINGLETON_CACHE["drift_detector"]


def get_prompt_manager(
    settings: Settings = Depends(get_app_settings),
) -> PromptManager:
    """Return versioned prompt manager."""
    if "prompt_manager" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["prompt_manager"] = PromptManager(settings)
    return _SINGLETON_CACHE["prompt_manager"]


def get_retriever(
    settings: Settings = Depends(get_app_settings),
) -> ChromaRetriever:
    """Return ChromaDB retriever."""
    if "retriever" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["retriever"] = ChromaRetriever(settings)
    return _SINGLETON_CACHE["retriever"]


def get_hallucination_detector() -> HallucinationDetector:
    """Return hallucination detector."""
    if "hallucination_detector" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["hallucination_detector"] = HallucinationDetector()
    return _SINGLETON_CACHE["hallucination_detector"]


def get_ragas_evaluator(
    settings: Settings = Depends(get_app_settings),
) -> RagasEvaluator:
    """Return Ragas evaluator."""
    if "ragas_evaluator" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["ragas_evaluator"] = RagasEvaluator(settings)
    return _SINGLETON_CACHE["ragas_evaluator"]


def get_conversation_memory(
    settings: Settings = Depends(get_app_settings),
) -> ConversationMemory:
    """Return conversation memory manager."""
    if "conversation_memory" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["conversation_memory"] = ConversationMemory(settings)
    return _SINGLETON_CACHE["conversation_memory"]


def get_rag_pipeline(
    retriever: ChromaRetriever = Depends(get_retriever),
    provider: BaseLLMProvider = Depends(get_llm_provider),
    drift: DriftDetector = Depends(get_drift_detector),
    prompts: PromptManager = Depends(get_prompt_manager),
    hallucination: HallucinationDetector = Depends(get_hallucination_detector),
    settings: Settings = Depends(get_app_settings),
) -> RAGPipeline:
    """Return the RAG pipeline."""
    if "rag_pipeline" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["rag_pipeline"] = RAGPipeline(
            retriever=retriever,
            llm_provider=provider,
            drift_detector=drift,
            prompt_manager=prompts,
            hallucination_detector=hallucination,
            settings=settings,
        )
    return _SINGLETON_CACHE["rag_pipeline"]


def get_document_ingestor(
    settings: Settings = Depends(get_app_settings),
) -> DocumentIngestor:
    """Return document ingestor."""
    if "document_ingestor" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["document_ingestor"] = DocumentIngestor(settings)
    return _SINGLETON_CACHE["document_ingestor"]


def get_file_chat_service(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    drift: DriftDetector = Depends(get_drift_detector),
    prompts: PromptManager = Depends(get_prompt_manager),
    settings: Settings = Depends(get_app_settings),
) -> FileChatService:
    """Return document and file-based chat service."""
    if "file_chat_service" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["file_chat_service"] = FileChatService(
            llm_provider=provider,
            drift_detector=drift,
            prompt_manager=prompts,
            settings=settings,
        )
    return _SINGLETON_CACHE["file_chat_service"]


def get_stt_provider(
    settings: Settings = Depends(get_app_settings),
) -> WhisperSTT:
    """Return Whisper speech-to-text provider."""
    if "stt_provider" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["stt_provider"] = WhisperSTT(settings)
    return _SINGLETON_CACHE["stt_provider"]


def get_tts_provider(
    settings: Settings = Depends(get_app_settings),
) -> GTTSTextToSpeech:
    """Return text-to-speech provider."""
    if "tts_provider" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["tts_provider"] = GTTSTextToSpeech(settings)
    return _SINGLETON_CACHE["tts_provider"]


def get_image_analyzer(
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> ImageAnalyzer:
    """Return image analyzer."""
    if "image_analyzer" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["image_analyzer"] = ImageAnalyzer(provider)
    return _SINGLETON_CACHE["image_analyzer"]


def get_image_generator(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    prompts: PromptManager = Depends(get_prompt_manager),
    settings: Settings = Depends(get_app_settings),
) -> ImageGenerator:
    """Return historical image generator."""
    if "image_generator" not in _SINGLETON_CACHE:
        _SINGLETON_CACHE["image_generator"] = ImageGenerator(
            llm_provider=provider, prompt_manager=prompts, settings=settings
        )
    return _SINGLETON_CACHE["image_generator"]


def get_video_generator(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    prompts: PromptManager = Depends(get_prompt_manager),
    settings: Settings = Depends(get_app_settings),
) -> Any:
    """Return historical video generator."""
    if "video_generator" not in _SINGLETON_CACHE:
        from agentverse.services.vision.video_generator import VideoGenerator

        _SINGLETON_CACHE["video_generator"] = VideoGenerator(
            llm_provider=provider, prompt_manager=prompts, settings=settings
        )
    return _SINGLETON_CACHE["video_generator"]


def get_agent_memory_manager(
    settings: Settings = Depends(get_app_settings),
) -> Any:
    """Return the isolated agent memory manager."""
    if "agent_memory_manager" not in _SINGLETON_CACHE:
        from agentverse.services.memory.agent_memory import AgentMemoryManager

        _SINGLETON_CACHE["agent_memory_manager"] = AgentMemoryManager(settings=settings)
    return _SINGLETON_CACHE["agent_memory_manager"]


def get_unlearning_engine(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    memory: Any = Depends(get_agent_memory_manager),
    retriever: ChromaRetriever = Depends(get_retriever),
) -> Any:
    """Return machine unlearning and validation engine."""
    if "unlearning_engine" not in _SINGLETON_CACHE:
        from agentverse.services.unlearning.engine import UnlearningEngine

        _SINGLETON_CACHE["unlearning_engine"] = UnlearningEngine(
            llm_provider=provider,
            memory_manager=memory,
            retriever=retriever,
        )
    return _SINGLETON_CACHE["unlearning_engine"]


def get_historyverse_ar_mode_agent(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    memory_manager: Any = Depends(get_agent_memory_manager),
) -> Any:
    """Return the AR Mode artifact guide agent instantiated from AgentTemplate."""
    if "ar_mode_agent" not in _SINGLETON_CACHE:
        from pathlib import Path

        from agentverse.services.agents.template import AgentTemplate, TemplatedAgent

        yaml_path = Path("configs/prompts/ar_mode.yaml")
        scoped_memory = memory_manager.get_agent_memory("ar-mode-agent")
        if yaml_path.exists():
            template = AgentTemplate.from_yaml(yaml_path)
        else:
            template = AgentTemplate(
                agent_id="ar-mode-agent",
                name="Mora",
                role="Augmented Reality Artifact Guide",
                domain="augmented-reality",
                system_prompt_template="Your name is {name}. You are an expert historical guide specialized in the monument or historical artifact: '{artifact_label}'.",
                default_variables={"artifact_label": "Ancient Artifact"},
            )
        _SINGLETON_CACHE["ar_mode_agent"] = TemplatedAgent(
            template=template,
            llm_provider=provider,
            memory=scoped_memory,
            metadata={"artifact_label": "Ancient Artifact", "agent_type": "ARMode"},
        )
    return _SINGLETON_CACHE["ar_mode_agent"]


get_ar_agent = get_historyverse_ar_mode_agent
