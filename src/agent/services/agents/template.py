"""Domain-Agnostic Agent Template and Factory.

Allows generating specialized domain agents with customizable personas, roles, prompts,
isolated memory, and distinct LLM provider engines. Supports loading directly from
declarative YAML template definitions.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from agent.core.base import BaseLLMProvider
from agent.services.agents.base import AgentResponse, BaseAgent
from agent.services.nlp.arabic import clean_arabic_text
from agent.services.nlp.colloquial import expand_egyptian_colloquial_query
from agent.services.rag.citation_guard import GroundingGuardrail
from agent.services.rag.refusal import DomainRefusalGate
from agent.services.security.pii import redact_egyptian_pii

if TYPE_CHECKING:
    from agent.services.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

__all__ = ["AgentTemplate", "TemplatedAgent"]


class AgentTemplate(BaseModel):
    """Declarative template schema for generating specialized domain agents.

    Decouples agent persona, role, prompt, domain, and provider configuration
    from any hardcoded application concept.
    """

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Persona name of the agent (e.g. 'Mora', 'Athena')")
    role: str = Field(
        ..., description="Role and job title (e.g. 'Historical Guide', 'AR Museum Tour Guide')"
    )
    domain: str = Field(
        default="general",
        description="Operational domain (e.g. 'history', 'augmented-reality', 'custom')",
    )
    system_prompt_template: str = Field(
        ..., description="Prompt template with optional variable placeholders"
    )
    default_variables: dict[str, Any] = Field(
        default_factory=dict, description="Default variables to interpolate into prompt"
    )
    provider_type: str | None = Field(
        default=None,
        description="Optional provider override ('gemini', 'openai', 'ollama', 'local')",
    )
    model_name: str | None = Field(default=None, description="Optional model identifier override")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> AgentTemplate:
        """Load an AgentTemplate definition directly from a YAML configuration file."""
        path = Path(yaml_path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        agent_id = data.get("agent_id") or data.get("name", "agent")
        name = data.get("persona_name") or data.get("assistant_name") or "Assistant"
        role = data.get("role") or data.get("description", "Specialized AI Agent")
        domain = data.get("domain", "general")
        system_prompt = data.get("template") or data.get("system_prompt", "")
        default_vars = data.get("default_variables", {})

        return cls(
            agent_id=agent_id,
            name=name,
            role=role,
            domain=domain,
            system_prompt_template=system_prompt,
            default_variables=default_vars,
            provider_type=data.get("provider_type"),
            model_name=data.get("model_name"),
            temperature=data.get("temperature"),
        )

    @classmethod
    def resolve(
        cls,
        template_name: str | None = None,
        prompts_dir: str | Path | None = None,
    ) -> AgentTemplate:
        """Dynamically resolve and instantiate an AgentTemplate by name without hardcoded paths."""
        from agent.config.settings import get_settings

        settings = get_settings()
        base_dir = Path(prompts_dir or settings.prompts_dir)
        target_name = template_name or settings.default_agent_template

        candidate_paths = [
            base_dir / f"{target_name}.yaml",
            base_dir / f"{target_name}.yml",
            base_dir / target_name,
            Path(target_name),
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return cls.from_yaml(candidate)

        return cls(
            agent_id=settings.agent_persona_id,
            name=settings.agent_persona_name,
            role="Specialized Domain Assistant",
            domain="general",
            system_prompt_template="Your name is {assistant_name}. You are an AI assistant specialized in analyzing questions accurately.",
        )


class TemplatedAgent(BaseAgent):
    """Concrete agent instantiated from an AgentTemplate."""

    def __init__(
        self,
        template: AgentTemplate,
        llm_provider: BaseLLMProvider,
        memory: AgentMemory | None = None,
        variable_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.template = template
        self.domain = template.domain
        self.variables = {**template.default_variables, **(variable_overrides or {})}
        self._custom_metadata = metadata or {}

        rendered_prompt = self._render_prompt()

        super().__init__(
            agent_id=template.agent_id,
            name=template.name,
            role=template.role,
            system_prompt=rendered_prompt,
            llm_provider=llm_provider,
            memory=memory,
        )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        llm_provider: BaseLLMProvider,
        memory: AgentMemory | None = None,
        variable_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemplatedAgent:
        """Instantiate a TemplatedAgent from a declarative YAML configuration file."""
        template = AgentTemplate.from_yaml(yaml_path)
        return cls(
            template=template,
            llm_provider=llm_provider,
            memory=memory,
            variable_overrides=variable_overrides,
            metadata=metadata,
        )

    @classmethod
    def create(
        cls,
        template_name: str | None = None,
        llm_provider: BaseLLMProvider | None = None,
        memory: AgentMemory | None = None,
        variable_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemplatedAgent:
        """Dynamically resolve template and instantiate TemplatedAgent without hardcoded paths."""
        from agent.config.settings import get_settings
        from agent.services.llm.factory import LLMFactory

        template = AgentTemplate.resolve(template_name)
        provider = llm_provider or LLMFactory.create(get_settings())
        return cls(
            template=template,
            llm_provider=provider,
            memory=memory,
            variable_overrides=variable_overrides,
            metadata=metadata,
        )

    def _render_prompt(self) -> str:
        prompt = self.template.system_prompt_template
        render_vars = {
            "name": self.template.name,
            "assistant_name": self.template.name,
            "role": self.template.role,
            "domain": self.domain,
            **self.variables,
        }
        for k, v in render_vars.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))
        return prompt

    def update_variables(self, **kwargs: Any) -> None:
        """Update template variables and regenerate effective prompt."""
        self.variables.update(kwargs)
        self.system_prompt = self._render_prompt()

    @property
    def artifact_label(self) -> str:
        """Retrieve active artifact or subject label for tour/inspection workflows."""
        return str(
            self.variables.get(
                "artifact_label",
                self._custom_metadata.get("artifact_label", "Ancient Artifact"),
            )
        )

    def set_artifact_label(self, label: str) -> None:
        """Dynamically scope agent persona and prompt to a specific artifact or monument."""
        self.update_variables(artifact_label=label)
        self._custom_metadata["artifact_label"] = label

    async def start_conversation(
        self,
        artifact_name: str | None = None,
        session_id: str | None = None,
        language: str = "auto",
    ) -> str:
        """Generate a concise multilingual overview of an artifact or subject (under 10 lines)."""
        from agent.utils.multilingual import detect_language, is_arabic

        label = artifact_name or self.artifact_label
        self.set_artifact_label(label)

        # Automatic language recognition if set to auto or unspecified
        target_lang = language
        if target_lang in ("auto", "unknown", ""):
            target_lang = "ar" if is_arabic(label) else detect_language(label)

        if target_lang == "ar":
            intro_query = f"قدم نبذة تاريخية شاملة وموجزة (في حدود 10 أسطر أو أقل) عن {label} وأهميتها التاريخية والأثرية."
        elif target_lang == "fr":
            intro_query = f"Donnez un aperçu historique complet et concis (en 10 lignes ou moins) sur {label} et son importance archéologique."
        elif target_lang == "de":
            intro_query = f"Geben Sie einen prägnanten historischen Überblick (in 10 Zeilen oder weniger) über {label} und seine archäologische Bedeutung."
        elif target_lang == "es":
            intro_query = f"Proporcione una descripción histórica concisa (en 10 líneas o menos) sobre {label} y su importancia arqueológica."
        elif target_lang == "it":
            intro_query = f"Fornisci una panoramica storica concisa (in 10 righe o meno) su {label} e la sua importanza archeologica."
        elif target_lang == "ru":
            intro_query = f"Предоставьте краткий исторический обзор (до 10 строк) о {label} и его археологическом значении."
        elif target_lang == "zh":
            intro_query = f"请简要介绍（10行以内）关于 {label} 的历史背景和考古价值。"
        else:
            intro_query = f"Tell me everything you can about {label}, but try to keep it brief (10 lines or less)."

        context: dict[str, Any] = {"session_id": session_id} if session_id else {}
        response = await self.run(intro_query, context=context)
        return response.output

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        ctx_data = context or {}
        context_str = ctx_data.get("context", "")
        session_id = ctx_data.get("session_id")

        # 1. Global Input Sanitization & Normalization
        sanitized_query = redact_egyptian_pii(query)
        normalized_query = clean_arabic_text(sanitized_query)
        enriched_query = expand_egyptian_colloquial_query(normalized_query)

        # 2. Global Domain Boundary & Scope Refusal Gate
        refusal_gate = DomainRefusalGate(
            domain=self.domain,
            artifact_label=self.artifact_label
            if self.domain in ("augmented-reality", "ar")
            else None,
            strict_mode=True,
        )
        decision = refusal_gate.check_query(normalized_query)
        if decision.is_refused:
            logger.info("Query refused by DomainRefusalGate: %s", decision.refusal_reason)
            return AgentResponse(
                agent_id=self.agent_id,
                agent_name=self.name,
                role=self.role,
                output=decision.refusal_message or "عذراً، هذا الاستفسار خارج نطاق تخصصي.",
                model_used=self.llm_provider.model_name,
                confidence=1.0,
                metadata={
                    "refusal": True,
                    "reason": decision.refusal_reason,
                    "domain": self.domain,
                },
            )

        history_str = (
            self.memory.get_formatted_history(session_id)
            if session_id and self.memory.get_history(session_id)
            else ""
        )

        # 3. Semantic Response Caching for Tourist Inquiries
        from agent.services.cache import get_semantic_cache

        cache = get_semantic_cache()
        cached_output = cache.get(
            query=normalized_query,
            domain=self.domain,
            artifact_label=self.artifact_label
            if self.domain in ("augmented-reality", "ar")
            else None,
        )
        if cached_output and not history_str:
            logger.info("Serving query from SemanticResponseCache (sub-15ms): %s", query[:50])
            return AgentResponse(
                agent_id=self.agent_id,
                agent_name=self.name,
                role=self.role,
                output=cached_output,
                model_used="semantic-cache",
                confidence=1.0,
                metadata={"cache_hit": True, "domain": self.domain, **self._custom_metadata},
            )

        system_base = self.build_effective_system_prompt()

        parts = [system_base]
        if history_str:
            parts.append(f"Conversation History:\n{history_str}")
        if context_str:
            parts.append(f"Context:\n{context_str}")
        parts.append(f"User Query:\n{enriched_query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info(
            "Executing templated agent [%s] in domain '%s' for query: %s",
            self.name,
            self.domain,
            query[:50],
        )
        response_text = await self.llm_provider.generate(formatted_prompt)

        # 4. Save to Semantic Response Cache
        if not history_str:
            cache.set(
                query=normalized_query,
                response=response_text,
                domain=self.domain,
                artifact_label=self.artifact_label
                if self.domain in ("augmented-reality", "ar")
                else None,
            )

        # 5. Post-Generation Fact & Source Guardrail
        if context_str:
            guardrail = GroundingGuardrail()
            verification = guardrail.verify_grounding(response_text, context_str)
            if not verification.is_valid:
                logger.warning(
                    "Unverified references detected: %s", verification.unverified_references
                )

        if session_id:
            self.memory.add_exchange(session_id, query, response_text)

        resp_metadata = {
            "domain": self.domain,
            **self._custom_metadata,
            **(context or {}),
        }

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output=response_text,
            model_used=self.llm_provider.model_name,
            confidence=0.95,
            metadata=resp_metadata,
        )

    async def stream(self, query: str, context: dict[str, Any] | None = None) -> AsyncIterator[str]:
        """Stream token responses asynchronously and record complete exchange into isolated memory."""
        ctx_data = context or {}
        context_str = ctx_data.get("context", "")
        session_id = ctx_data.get("session_id")

        # 1. Global Input Sanitization & Normalization
        sanitized_query = redact_egyptian_pii(query)
        normalized_query = clean_arabic_text(sanitized_query)
        enriched_query = expand_egyptian_colloquial_query(normalized_query)

        # 2. Global Domain Boundary & Scope Refusal Gate
        refusal_gate = DomainRefusalGate(
            domain=self.domain,
            artifact_label=self.artifact_label
            if self.domain in ("augmented-reality", "ar")
            else None,
            strict_mode=True,
        )
        decision = refusal_gate.check_query(normalized_query)
        if decision.is_refused:
            logger.info("Stream refused by DomainRefusalGate: %s", decision.refusal_reason)
            yield decision.refusal_message or "عذراً، هذا الاستفسار خارج نطاق تخصصي."
            return

        history_str = (
            self.memory.get_formatted_history(session_id)
            if session_id and self.memory.get_history(session_id)
            else ""
        )
        system_base = self.build_effective_system_prompt()

        parts = [system_base]
        if history_str:
            parts.append(f"Conversation History:\n{history_str}")
        if context_str:
            parts.append(f"Context:\n{context_str}")
        parts.append(f"User Query:\n{enriched_query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info(
            "Streaming templated agent [%s] in domain '%s' for query: %s",
            self.name,
            self.domain,
            query[:50],
        )
        accumulated_chunks: list[str] = []
        try:
            async for token in self.llm_provider.stream(formatted_prompt):
                accumulated_chunks.append(token)
                yield token
        finally:
            full_answer = "".join(accumulated_chunks).strip()
            if session_id and full_answer:
                self.memory.add_exchange(session_id, query, full_answer)
