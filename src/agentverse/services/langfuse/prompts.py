"""Prompt versioning and template management service."""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from agentverse.config.settings import Settings, get_settings
from agentverse.services.langfuse.client import get_langfuse_client

logger = logging.getLogger(__name__)

__all__ = ["PromptManager"]


class PromptManager:
    """Manages versioned prompts from Langfuse with fallback to local YAML templates."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._prompts_dir = self._settings.prompts_dir
        self._cache: dict[str, str] = {}

    def get_prompt(
        self,
        name: str = "historical_chat",
        version: str | None = None,
    ) -> str:
        """Retrieve the prompt template string by name and version."""
        ver = version or self._settings.langfuse_system_prompt_version
        cache_key = f"{name}:{ver}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        client = get_langfuse_client(self._settings)
        if client:
            try:
                prompt_obj = client.get_prompt(name, version=ver)
                template = prompt_obj.compile()
                self._cache[cache_key] = template
                logger.info("Loaded prompt '%s' version '%s' from Langfuse", name, ver)
                return template
            except Exception as exc:
                logger.warning(
                    "Failed to fetch prompt '%s' from Langfuse: %s. Using local fallback.",
                    name,
                    exc,
                )

        local_template = self._load_from_yaml(name)
        self._cache[cache_key] = local_template
        return local_template

    def _load_from_yaml(self, name: str) -> str:
        yaml_path = os.path.join(self._prompts_dir, f"{name}.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    return data.get("template", "")
            except Exception as exc:
                logger.error("Error reading local prompt file %s: %s", yaml_path, exc)

        return (
            "Your name is {assistant_name}. You are an expert AI assistant specializing in "
            "domain analysis and accurate question answering.\n"
            "Context: {context}\n"
            "History: {history}\n"
            "Question: {question}\n"
            "Answer:"
        )

    def render(self, name: str, version: str | None = None, **kwargs: Any) -> str:
        """Render a versioned prompt with provided variable substitutions."""
        template = self.get_prompt(name, version=version)
        persona_name = (
            kwargs.get("assistant_name")
            or kwargs.get("name")
            or kwargs.get("persona_name")
            or self._settings.agent_persona_name
        )
        model_name = (
            kwargs.get("model_name") or kwargs.get("model") or self._settings.llm_model_name
        )
        kwargs.setdefault("assistant_name", persona_name)
        kwargs.setdefault("name", persona_name)
        kwargs.setdefault("persona_name", persona_name)
        kwargs.setdefault("model_name", model_name)
        kwargs.setdefault("model", model_name)
        try:
            return template.format(**kwargs)
        except KeyError:
            for k, v in kwargs.items():
                template = template.replace(f"{{{k}}}", str(v))
            return template
