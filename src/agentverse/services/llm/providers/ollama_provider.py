"""Ollama LLM provider with automated pulling, health probes, and multimodal vision support."""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import urllib.error
import urllib.request
from typing import Any

import httpx
import litellm
from PIL import Image

from agentverse.config.settings import Settings
from agentverse.core.base import BaseLLMProvider
from agentverse.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

__all__ = ["OllamaProvider"]


class OllamaProvider(BaseLLMProvider):
    """First-class provider for Ollama running locally or in a remote container."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        raw_name = settings.llm_model_name
        self._model_name = (
            raw_name.replace("ollama/", "") if raw_name.startswith("ollama/") else raw_name
        )
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._auto_download = settings.llm_auto_download
        self._ensure_model_available()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model_available(self) -> None:
        """Verify model availability in Ollama; auto-pull via API or CLI if enabled."""
        if not self._auto_download:
            return

        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                clean_target = self._model_name.split(":")[0]
                if clean_target in models:
                    logger.info("Ollama model '%s' already cached and ready.", self._model_name)
                    return
        except Exception as exc:
            logger.debug("Ollama tags probe skipped: %s", exc)

        logger.info("Auto-pulling Ollama model '%s' via API...", self._model_name)
        try:
            pull_url = f"{self._base_url}/api/pull"
            payload = json.dumps({"name": self._model_name, "stream": False}).encode("utf-8")
            req = urllib.request.Request(
                pull_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                logger.info(
                    "Ollama model '%s' pulled successfully (HTTP %d).",
                    self._model_name,
                    resp.status,
                )
                return
        except Exception as api_exc:
            logger.warning("Ollama API pull failed (%s); trying CLI fallback...", api_exc)

        try:
            subprocess.run(
                ["ollama", "pull", self._model_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as cli_exc:
            logger.warning("Ollama CLI pull fallback skipped: %s", cli_exc)

    async def is_available(self) -> bool:
        """Probe Ollama server readiness."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self._base_url}/api/version")
                return res.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        *,
        images: list[Image.Image] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response via LiteLLM with direct Ollama REST fallback."""
        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        litellm_model = f"ollama/{self._model_name}"

        # If multimodal images are provided, convert to base64 for Ollama vision
        encoded_images: list[str] = []
        if images:
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                encoded_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        if not encoded_images:
            try:
                response = await litellm.acompletion(
                    model=litellm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=tokens,
                    api_base=self._base_url,
                    **kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning(
                    "LiteLLM Ollama call failed (%s); attempting direct REST fallback...", exc
                )

        # Direct REST API fallback (supports vision models e.g. llava, llama3.2-vision)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload: dict[str, Any] = {
                    "model": self._model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": tokens,
                    },
                }
                if encoded_images:
                    payload["images"] = encoded_images

                res = await client.post(f"{self._base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "").strip()
        except Exception as exc:
            logger.error("Direct Ollama API generation failed: %s", exc)
            raise LLMProviderError(f"Ollama generation failed: {exc}", provider="ollama") from exc
