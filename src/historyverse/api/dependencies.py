"""FastAPI dependency injection container."""

from __future__ import annotations

import logging
from functools import lru_cache
from fastapi import Depends

from historyverse.config.settings import Settings, get_settings
from historyverse.core.base import BaseLLMProvider
from historyverse.services.evaluation.hallucination import HallucinationDetector
from historyverse.services.evaluation.ragas_eval import RagasEvaluator
from historyverse.services.langfuse.prompts import PromptManager
from historyverse.services.llm.drift import DriftDetector
from historyverse.services.llm.factory import LLMFactory
from historyverse.services.memory.manager import ConversationMemory
from historyverse.services.rag.ingest import DocumentIngestor
from historyverse.services.rag.pipeline import RAGPipeline
from historyverse.services.rag.retriever import ChromaRetriever
from historyverse.services.vision.analyzer import ImageAnalyzer
from historyverse.services.vision.generator import ImageGenerator
from historyverse.services.voice.stt import WhisperSTT
from historyverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)

__all__ = [
    "get_app_settings",
    "get_conversation_memory",
    "get_document_ingestor",
    "get_drift_detector",
    "get_hallucination_detector",
    "get_image_analyzer",
    "get_image_generator",
    "get_llm_provider",
    "get_prompt_manager",
    "get_rag_pipeline",
    "get_ragas_evaluator",
    "get_retriever",
    "get_stt_provider",
    "get_tts_provider",
]


def get_app_settings() -> Settings:
    """Return application settings."""
    return get_settings()


@lru_cache(maxsize=1)
def get_llm_provider(settings: Settings = Depends(get_app_settings)) -> BaseLLMProvider:
    """Return the configured primary LLM provider."""
    return LLMFactory.create(settings)


@lru_cache(maxsize=1)
def get_drift_detector(
    settings: Settings = Depends(get_app_settings),
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> DriftDetector:
    """Return the drift detector configured with primary and fallback providers."""
    fallback_provider = None
    if settings.drift_fallback_model:
        try:
            fallback_provider = LLMFactory.create(settings)
        except Exception as exc:
            logger.warning("Could not instantiate fallback provider: %s", exc)

    return DriftDetector(
        primary_provider=provider,
        fallback_provider=fallback_provider,
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_prompt_manager(
    settings: Settings = Depends(get_app_settings),
) -> PromptManager:
    """Return versioned prompt manager."""
    return PromptManager(settings)


@lru_cache(maxsize=1)
def get_retriever(
    settings: Settings = Depends(get_app_settings),
) -> ChromaRetriever:
    """Return ChromaDB retriever."""
    return ChromaRetriever(settings)


@lru_cache(maxsize=1)
def get_hallucination_detector() -> HallucinationDetector:
    """Return hallucination detector."""
    return HallucinationDetector()


@lru_cache(maxsize=1)
def get_ragas_evaluator(
    settings: Settings = Depends(get_app_settings),
) -> RagasEvaluator:
    """Return Ragas evaluator."""
    return RagasEvaluator(settings)


@lru_cache(maxsize=1)
def get_conversation_memory(
    settings: Settings = Depends(get_app_settings),
) -> ConversationMemory:
    """Return conversation memory manager."""
    return ConversationMemory(settings)


@lru_cache(maxsize=1)
def get_rag_pipeline(
    retriever: ChromaRetriever = Depends(get_retriever),
    provider: BaseLLMProvider = Depends(get_llm_provider),
    drift: DriftDetector = Depends(get_drift_detector),
    prompts: PromptManager = Depends(get_prompt_manager),
    hallucination: HallucinationDetector = Depends(get_hallucination_detector),
    settings: Settings = Depends(get_app_settings),
) -> RAGPipeline:
    """Return the RAG pipeline."""
    return RAGPipeline(
        retriever=retriever,
        llm_provider=provider,
        drift_detector=drift,
        prompt_manager=prompts,
        hallucination_detector=hallucination,
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_document_ingestor(
    settings: Settings = Depends(get_app_settings),
) -> DocumentIngestor:
    """Return document ingestor."""
    return DocumentIngestor(settings)


@lru_cache(maxsize=1)
def get_stt_provider(
    settings: Settings = Depends(get_app_settings),
) -> WhisperSTT:
    """Return Whisper speech-to-text provider."""
    return WhisperSTT(settings)


@lru_cache(maxsize=1)
def get_tts_provider(
    settings: Settings = Depends(get_app_settings),
) -> GTTSTextToSpeech:
    """Return text-to-speech provider."""
    return GTTSTextToSpeech(settings)


def get_image_analyzer(
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> ImageAnalyzer:
    """Return image analyzer."""
    return ImageAnalyzer(provider)


def get_image_generator(
    provider: BaseLLMProvider = Depends(get_llm_provider),
    prompts: PromptManager = Depends(get_prompt_manager),
    settings: Settings = Depends(get_app_settings),
) -> ImageGenerator:
    """Return historical image generator."""
    return ImageGenerator(llm_provider=provider, prompt_manager=prompts, settings=settings)
