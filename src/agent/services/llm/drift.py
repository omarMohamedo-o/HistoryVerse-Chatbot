"""Drift detection and automated re-run execution engine."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from PIL import Image

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseLLMProvider
from agent.core.enums import DriftAction
from agent.core.schemas import DriftReport

logger = logging.getLogger(__name__)

__all__ = ["DriftDetector"]


class DriftDetector:
    """Monitors model output confidence, triggers re-runs, and dispatches drift actions."""

    def __init__(
        self,
        primary_provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._primary = primary_provider
        self._fallback = fallback_provider
        self._threshold = self._settings.drift_confidence_threshold
        self._max_retries = self._settings.drift_max_retries
        self._action = self._settings.drift_action

    async def generate_with_drift_protection(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        **kwargs: Any,
    ) -> tuple[str, DriftReport]:
        """Generate response with drift monitoring and automated retry execution."""
        first_answer, first_conf = await self._primary.generate_with_confidence(
            prompt, images=images, **kwargs
        )

        if first_conf >= self._threshold:
            report = DriftReport(
                confidence_score=first_conf,
                threshold=self._threshold,
                action_taken=DriftAction.RETRY,
                original_answer=first_answer,
                final_answer=first_answer,
                retries=0,
                fallback_used=False,
                docker_rebuild_triggered=False,
            )
            return first_answer, report

        logger.warning(
            "Model drift detected on initial output: conf=%.2f < threshold=%.2f. Triggering re-run.",
            first_conf,
            self._threshold,
        )

        best_answer = first_answer
        best_conf = first_conf
        retry_count = 0

        while retry_count < self._max_retries and best_conf < self._threshold:
            retry_count += 1
            adjusted_temp = min(1.0, 0.2 + (0.15 * retry_count))
            logger.info(
                "Executing retry attempt %d/%d with adjusted temperature=%.2f",
                retry_count,
                self._max_retries,
                adjusted_temp,
            )

            alt_answer, alt_conf = await self._primary.generate_with_confidence(
                prompt, images=images, temperature=adjusted_temp, **kwargs
            )
            if alt_conf > best_conf:
                best_answer = alt_answer
                best_conf = alt_conf

        fallback_used = False
        if best_conf < self._threshold and self._fallback is not None:
            logger.warning(
                "Retries exhausted. Activating fallback provider=%s",
                self._fallback.model_name,
            )
            try:
                fallback_ans, fallback_conf = await self._fallback.generate_with_confidence(
                    prompt, images=images, **kwargs
                )
                if fallback_conf >= best_conf:
                    best_answer = fallback_ans
                    best_conf = fallback_conf
                    fallback_used = True
            except Exception as exc:
                logger.error("Fallback generation failed: %s", exc)

        docker_triggered = False
        if best_conf < self._threshold:
            docker_triggered = self._handle_persistent_drift()

        report = DriftReport(
            confidence_score=best_conf,
            threshold=self._threshold,
            action_taken=self._action,
            original_answer=first_answer,
            final_answer=best_answer,
            retries=retry_count,
            fallback_used=fallback_used,
            docker_rebuild_triggered=docker_triggered,
        )
        return best_answer, report

    def trigger_docker_rebuild(self) -> bool:
        """Trigger container rebuild command."""
        cmd = self._settings.docker_rebuild_command
        logger.warning("Executing container rebuild command: %s", cmd)
        try:
            subprocess.Popen(cmd, shell=True)
            return True
        except Exception as exc:
            logger.error("Failed to execute docker rebuild command: %s", exc)
            return False

    def trigger_reingest(self) -> bool:
        """Trigger automatic document re-ingestion to fix context drift."""
        try:
            from agent.services.rag.ingest import DocumentIngestor

            logger.info("Executing automatic document re-ingestion to resolve knowledge drift...")
            ingestor = DocumentIngestor(self._settings)
            ingestor.ingest()
            return True
        except Exception as exc:
            logger.error("Automated re-ingestion failed: %s", exc)
            return False

    def _handle_persistent_drift(self) -> bool:
        """Execute configured policy when drift persists beyond retries and fallback."""
        if self._action == DriftAction.DOCKER_REBUILD:
            return self.trigger_docker_rebuild()

        logger.warning(
            "Persistent drift recorded: threshold=%.2f action=%s",
            self._threshold,
            self._action,
        )
        return False
