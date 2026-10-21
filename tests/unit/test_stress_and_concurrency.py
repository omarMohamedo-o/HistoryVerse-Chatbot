"""Stress, high concurrency, and provider outage resilience tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config.settings import Settings
from agent.services.agents.domain import DomainAgent
from agent.services.llm.drift import DriftDetector
from agent.services.memory.manager import ConversationMemory


@pytest.mark.asyncio
async def test_high_concurrency_agent_execution():
    """Simulate 30 concurrent agent executions to ensure no race conditions or thread deadlocks."""
    mock_provider = AsyncMock()
    mock_provider.model_name = "stress-mock-model"
    mock_provider.generate = AsyncMock(
        side_effect=lambda q, **kwargs: f"Concurrently processed: {q[:10]}"
    )
    mock_provider.estimate_confidence = MagicMock(return_value=0.95)

    agent = DomainAgent(
        agent_id="stress-tester",
        name="Concurrent Worker",
        role="Load Tester",
        system_prompt="Execute tasks concurrently.",
        llm_provider=mock_provider,
        domain="testing",
    )

    queries = [f"Query batch request #{i} for stress analysis" for i in range(30)]

    async def execute_task(q: str):
        res = await agent.run(q)
        return res

    tasks = [execute_task(q) for q in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 30
    for res in results:
        assert res.agent_id == "stress-tester"
        assert "Concurrently processed:" in res.output
    assert mock_provider.generate.call_count == 30


@pytest.mark.asyncio
async def test_concurrent_conversation_memory_access():
    """Simulate 25 concurrent users writing and reading distinct conversation sessions."""
    settings = Settings()
    memory = ConversationMemory(settings)

    async def user_session_flow(user_id: int):
        session_id = f"user_sess_{user_id}"
        await memory.add_exchange_async(
            session_id, f"Hello from user {user_id}", f"Reply to user {user_id}"
        )
        history = await memory.get_formatted_history_async(session_id)
        assert f"Hello from user {user_id}" in history
        assert f"Reply to user {user_id}" in history
        return session_id

    tasks = [user_session_flow(i) for i in range(25)]
    completed = await asyncio.gather(*tasks)
    assert len(completed) == 25


@pytest.mark.asyncio
async def test_provider_outage_and_fallback_recovery():
    """Simulate primary LLM provider low-quality/outage output triggering fallback recovery."""
    settings = Settings(drift_threshold=0.7)

    # Primary provider produces sub-threshold / degraded confidence
    degraded_primary = AsyncMock()
    degraded_primary.model_name = "primary-model"
    degraded_primary.generate_with_confidence = AsyncMock(
        return_value=("Degraded output due to transient upstream congestion", 0.35)
    )

    # Fallback provider succeeds with high confidence
    healthy_fallback = AsyncMock()
    healthy_fallback.model_name = "fallback-model"
    healthy_fallback.generate_with_confidence = AsyncMock(
        return_value=("Recovered response from fallback service", 0.94)
    )

    detector = DriftDetector(
        primary_provider=degraded_primary,
        fallback_provider=healthy_fallback,
        settings=settings,
    )

    answer, report = await detector.generate_with_drift_protection("Critical historical query")

    assert answer == "Recovered response from fallback service"
    assert report.confidence_score == 0.94
    assert report.fallback_used is True
    assert degraded_primary.generate_with_confidence.call_count >= 1
    assert healthy_fallback.generate_with_confidence.call_count == 1
