"""Domain-Agnostic Agent Template and Factory.

Allows generating specialized domain agents (such as HistoryVerse, AR Mode,
or any arbitrary domain agent) with customizable personas, roles, prompts,
isolated memory, and distinct LLM provider engines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from historyverse.core.base import BaseLLMProvider
from historyverse.services.agents.base import AgentResponse, BaseAgent

if TYPE_CHECKING:
    from historyverse.services.memory.agent_memory import AgentMemory

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

    def _render_prompt(self) -> str:
        prompt = self.template.system_prompt_template
        render_vars = {
            "name": self.template.name,
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

    async def run(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        ctx_data = context or {}
        context_str = ctx_data.get("context", "")
        session_id = ctx_data.get("session_id")
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
        parts.append(f"User Query:\n{query}")
        formatted_prompt = "\n\n".join(parts)

        logger.info(
            "Executing templated agent [%s] in domain '%s' for query: %s",
            self.name,
            self.domain,
            query[:50],
        )
        response_text = await self.llm_provider.generate(formatted_prompt)

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
